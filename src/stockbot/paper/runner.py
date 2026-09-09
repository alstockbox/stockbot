from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from stockbot.arena.experiments import _signal_weights
from stockbot.data.panel import build_panel
from stockbot.data.point_in_time_features import PointInTimeFeatureStore
from stockbot.domain.models import TargetPosition
from stockbot.features.research_matrix import build_research_feature_matrix
from stockbot.ml.labels import make_panel_labels
from stockbot.ml.models import ModelConfig, build_model
from stockbot.paper.ledger import PaperObservation, PaperTradingLedger, make_paper_observation
from stockbot.research.liquidity_execution import LiquidityExecutionConfig
from stockbot.research.quarantine_audit import FrozenStrategySpec
from stockbot.risk.engine import RiskConfig, RiskEngine, RiskState


@dataclass(frozen=True)
class FrozenShadowArtifactManifest:
    artifact_id: str
    strategy_id: str
    experiment_id: str
    research_cycle_id: str
    source_dataset_fingerprint: str
    horizon: int
    model_name: str
    model_params: dict[str, Any]
    seed: int
    top_fraction: float
    weighting: str
    feature_names: tuple[str, ...]
    symbols: tuple[str, ...]
    freeze_timestamp: str
    last_training_label_timestamp: str
    model_filename: str
    model_sha256: str
    training_matrix_fingerprint: str
    execution_config: dict[str, Any]
    risk_config: dict[str, Any]
    auxiliary_fingerprint: str | None
    frozen_at: str
    research_readiness_fingerprint: str | None = None
    broker_execution_available: bool = False
    schema_version: int = 1


@dataclass(frozen=True)
class PaperRunnerState:
    strategy_id: str
    artifact_id: str
    last_processed_timestamp: str | None
    current_weights: dict[str, float]
    pending_signal_timestamp: str | None
    pending_target_weights: dict[str, float]
    pending_signal_snapshot_fingerprint: str | None
    equity: float = 100.0
    equity_peak: float = 100.0
    schema_version: int = 1


@dataclass(frozen=True)
class ShadowStepResult:
    strategy_id: str
    artifact_id: str
    processed_timestamp: str
    pending_signal_timestamp: str
    target_weights: dict[str, float]
    observation: PaperObservation | None
    idempotent_replay: bool = False
    broker_execution_available: bool = False


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _training_matrix_fingerprint(features: pd.DataFrame, labels: pd.Series) -> str:
    stable = pd.concat([features, labels.rename("__label__")], axis=1)
    h = hashlib.sha256()
    h.update(json.dumps([str(value) for value in stable.columns], separators=(",", ":")).encode("utf-8"))
    h.update(json.dumps([str(value) for value in stable.dtypes], separators=(",", ":")).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(stable, index=True, categorize=False).to_numpy().tobytes())
    return h.hexdigest()


def _manifest_identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_id"}
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalize_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["feature_names"] = tuple(normalized.get("feature_names", ()))
    normalized["symbols"] = tuple(normalized.get("symbols", ()))
    normalized["model_params"] = dict(normalized.get("model_params", {}))
    normalized["execution_config"] = dict(normalized.get("execution_config", {}))
    normalized["risk_config"] = dict(normalized.get("risk_config", {}))
    return normalized


def freeze_shadow_strategy(
    bars: pd.DataFrame,
    spec: FrozenStrategySpec,
    *,
    feature_names: tuple[str, ...] | list[str],
    research_cycle_id: str,
    source_dataset_fingerprint: str,
    output_dir: str | Path,
    research_readiness_fingerprint: str | None = None,
    execution_config: LiquidityExecutionConfig | None = None,
    risk_config: RiskConfig | None = None,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> FrozenShadowArtifactManifest:
    """Fit one immutable model for genuine forward shadow evaluation.

    The final model is fitted once using only rows whose forward label is fully known in
    the freeze dataset. Daily shadow steps load this model; they never retrain it.
    """

    cycle_id = str(research_cycle_id).strip()
    dataset_id = str(source_dataset_fingerprint).strip()
    readiness_id = (
        None
        if research_readiness_fingerprint is None
        else str(research_readiness_fingerprint).strip()
    )
    selected_features = tuple(str(value).strip() for value in feature_names if str(value).strip())
    if not cycle_id:
        raise ValueError("research_cycle_id is required")
    if not dataset_id:
        raise ValueError("source_dataset_fingerprint is required")
    if research_readiness_fingerprint is not None and not readiness_id:
        raise ValueError("research_readiness_fingerprint cannot be empty")
    if not selected_features or len(set(selected_features)) != len(selected_features):
        raise ValueError("feature_names must contain unique features")

    panel = build_panel(bars)
    matrix = build_research_feature_matrix(
        panel,
        feature_columns=selected_features,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_feature_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )
    labels = make_panel_labels(panel, horizons=(spec.horizon,))[f"fwd_return_{spec.horizon}"]
    labels.name = f"fwd_return_{spec.horizon}"
    valid = matrix.frame.notna().all(axis=1) & labels.notna() & np.isfinite(labels.astype(float))
    if int(valid.sum()) < 10:
        raise ValueError("insufficient fully-known training rows to freeze shadow model")

    model = build_model(ModelConfig(spec.model_name, params=dict(spec.model_params), seed=int(spec.seed)))
    model.fit(
        matrix.frame.loc[valid].to_numpy(dtype=float),
        labels.loc[valid].to_numpy(dtype=float),
    )

    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    model_sha = _sha256_bytes(model_bytes)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_filename = "model.pkl"
    model_path = target_dir / model_filename
    temporary_model = target_dir / (model_filename + ".tmp")
    temporary_model.write_bytes(model_bytes)
    temporary_model.replace(model_path)

    timestamps = panel.index.get_level_values("timestamp")
    valid_timestamps = matrix.frame.loc[valid].index.get_level_values("timestamp")
    execution = execution_config or LiquidityExecutionConfig()
    risk = risk_config or RiskConfig()
    auxiliary_fingerprint = None if auxiliary_store is None else auxiliary_store.fingerprint
    payload: dict[str, Any] = {
        "artifact_id": "",
        "strategy_id": str(spec.strategy_id),
        "experiment_id": str(spec.experiment_id),
        "research_cycle_id": cycle_id,
        "source_dataset_fingerprint": dataset_id,
        "horizon": int(spec.horizon),
        "model_name": str(spec.model_name),
        "model_params": dict(spec.model_params),
        "seed": int(spec.seed),
        "top_fraction": float(spec.top_fraction),
        "weighting": str(spec.weighting),
        "feature_names": list(selected_features),
        "symbols": sorted(str(value) for value in panel.index.get_level_values("symbol").unique()),
        "freeze_timestamp": pd.Timestamp(timestamps.max()).isoformat(),
        "last_training_label_timestamp": pd.Timestamp(valid_timestamps.max()).isoformat(),
        "model_filename": model_filename,
        "model_sha256": model_sha,
        "training_matrix_fingerprint": _training_matrix_fingerprint(matrix.frame.loc[valid], labels.loc[valid]),
        "execution_config": asdict(execution),
        "risk_config": asdict(risk),
        "auxiliary_fingerprint": auxiliary_fingerprint,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "research_readiness_fingerprint": readiness_id,
        "broker_execution_available": False,
        "schema_version": 1,
    }
    payload["artifact_id"] = _stable_json_hash(_manifest_identity_payload(payload))[:24]
    _write_json_atomic(target_dir / "manifest.json", payload)
    return FrozenShadowArtifactManifest(**_normalize_manifest_payload(payload))


def _load_manifest(artifact_dir: str | Path) -> FrozenShadowArtifactManifest:
    target_dir = Path(artifact_dir)
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("frozen shadow artifact manifest is missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen shadow artifact manifest must be a JSON object")
    normalized = _normalize_manifest_payload(payload)
    manifest = FrozenShadowArtifactManifest(**normalized)
    expected_id = _stable_json_hash(_manifest_identity_payload(payload))[:24]
    if manifest.artifact_id != expected_id:
        raise ValueError("frozen shadow artifact manifest fingerprint mismatch")
    if manifest.broker_execution_available:
        raise ValueError("frozen shadow artifact may not enable broker execution")
    return manifest


def _verified_model_bytes(artifact_dir: str | Path, manifest: FrozenShadowArtifactManifest) -> bytes:
    filename = Path(manifest.model_filename)
    if filename.name != manifest.model_filename or filename.is_absolute():
        raise ValueError("invalid frozen model filename")
    model_path = Path(artifact_dir) / filename
    if not model_path.exists():
        raise ValueError("frozen shadow model file is missing")
    payload = model_path.read_bytes()
    if _sha256_bytes(payload) != manifest.model_sha256:
        raise ValueError("frozen shadow model hash mismatch")
    return payload


def load_frozen_shadow_artifact(artifact_dir: str | Path) -> FrozenShadowArtifactManifest:
    """Verify manifest identity and model bytes without deserializing the model."""

    manifest = _load_manifest(artifact_dir)
    _verified_model_bytes(artifact_dir, manifest)
    return manifest


def _load_verified_model(artifact_dir: str | Path, manifest: FrozenShadowArtifactManifest):
    payload = _verified_model_bytes(artifact_dir, manifest)
    # Only artifact bytes whose SHA-256 is bound into the verified local manifest are
    # deserialized. Callers must not point this loader at untrusted external artifacts.
    return pickle.loads(payload)


def _load_state(path: Path, manifest: FrozenShadowArtifactManifest) -> PaperRunnerState:
    if not path.exists():
        return PaperRunnerState(
            strategy_id=manifest.strategy_id,
            artifact_id=manifest.artifact_id,
            last_processed_timestamp=None,
            current_weights={symbol: 0.0 for symbol in manifest.symbols},
            pending_signal_timestamp=None,
            pending_target_weights={},
            pending_signal_snapshot_fingerprint=None,
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("paper runner state must be a JSON object")
    payload["current_weights"] = {str(key): float(value) for key, value in payload.get("current_weights", {}).items()}
    payload["pending_target_weights"] = {
        str(key): float(value) for key, value in payload.get("pending_target_weights", {}).items()
    }
    state = PaperRunnerState(**payload)
    if state.strategy_id != manifest.strategy_id or state.artifact_id != manifest.artifact_id:
        raise ValueError("paper runner state does not match frozen strategy artifact")
    return state


def _save_state(path: Path, state: PaperRunnerState) -> None:
    _write_json_atomic(path, asdict(state))


def _append_or_verify_observation(
    ledger: PaperTradingLedger,
    observation: PaperObservation,
) -> PaperObservation:
    existing = [
        row
        for row in ledger.records(strategy_id=observation.strategy_id)
        if row.timestamp == observation.timestamp
    ]
    if not existing:
        ledger.append(observation)
        return observation
    if len(existing) != 1 or existing[0] != observation:
        raise ValueError("existing paper observation conflicts with recovered shadow settlement")
    return existing[0]


def _paper_risk_state(
    ledger: PaperTradingLedger,
    strategy_id: str,
    *,
    data_age_seconds: float,
    kill_switch: bool,
) -> RiskState:
    rows = ledger.records(strategy_id=strategy_id)
    equity = 100.0
    peak = 100.0
    recent_returns: list[float] = []
    for row in rows:
        equity *= 1.0 + float(row.net_return)
        peak = max(peak, equity)
        recent_returns.append(float(row.net_return))
    recent = np.asarray(recent_returns[-20:], dtype=float)
    realized_vol = float(np.std(recent, ddof=0) * np.sqrt(252.0)) if len(recent) >= 2 else 0.0
    daily_pnl = float(rows[-1].net_return) if rows else 0.0
    return RiskState(
        equity=equity,
        equity_peak=peak,
        daily_pnl_pct=daily_pnl,
        data_age_seconds=float(data_age_seconds),
        realized_volatility=realized_vol,
        kill_switch=bool(kill_switch),
    )


def _risk_adjust_targets(
    targets: dict[str, float],
    manifest: FrozenShadowArtifactManifest,
    state: RiskState,
) -> dict[str, float]:
    engine = RiskEngine(RiskConfig(**manifest.risk_config))
    approved: dict[str, float] = {}
    for symbol in manifest.symbols:
        requested = max(0.0, min(1.0, float(targets.get(symbol, 0.0))))
        decision = engine.evaluate(TargetPosition(symbol, requested), state)
        approved[symbol] = float(decision.target.weight)
    return approved


def _latest_predictions(
    bars: pd.DataFrame,
    manifest: FrozenShadowArtifactManifest,
    model,
    *,
    auxiliary_store: PointInTimeFeatureStore | None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None,
    auxiliary_max_age_days: int | None,
    auxiliary_min_coverage: float,
) -> tuple[pd.Timestamp, pd.Series]:
    panel = build_panel(bars)
    symbols = set(str(value) for value in panel.index.get_level_values("symbol").unique())
    if symbols != set(manifest.symbols):
        raise ValueError("paper snapshot symbol universe does not match frozen artifact")
    if manifest.auxiliary_fingerprint is not None:
        if auxiliary_store is None or auxiliary_store.fingerprint != manifest.auxiliary_fingerprint:
            raise ValueError("paper auxiliary feature store does not match frozen artifact")
    elif auxiliary_store is not None and any(name.startswith("aux__") for name in manifest.feature_names):
        raise ValueError("frozen auxiliary feature contract is missing its fingerprint")

    matrix = build_research_feature_matrix(
        panel,
        feature_columns=manifest.feature_names,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_feature_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )
    latest = pd.Timestamp(panel.index.get_level_values("timestamp").max())
    latest_rows = matrix.frame.xs(latest, level="timestamp", drop_level=False)
    valid = latest_rows.notna().all(axis=1)
    predictions = pd.Series(np.nan, index=latest_rows.index, dtype=float, name="shadow_prediction")
    if int(valid.sum()) > 0:
        values = np.asarray(model.predict(latest_rows.loc[valid].to_numpy(dtype=float)), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("frozen shadow model produced non-finite predictions")
        predictions.loc[latest_rows.loc[valid].index] = values
    return latest, predictions


def _settle_pending_session(
    bars: pd.DataFrame,
    manifest: FrozenShadowArtifactManifest,
    state: PaperRunnerState,
    *,
    realization_timestamp: pd.Timestamp,
    realization_snapshot_fingerprint: str,
) -> tuple[PaperObservation, dict[str, float], float, float]:
    if state.pending_signal_timestamp is None or state.pending_signal_snapshot_fingerprint is None:
        raise ValueError("pending paper signal provenance is incomplete")
    signal_timestamp = pd.Timestamp(state.pending_signal_timestamp)
    if signal_timestamp.tzinfo is None:
        signal_timestamp = signal_timestamp.tz_localize("UTC")
    if realization_timestamp <= signal_timestamp:
        raise ValueError("paper realization must occur after pending signal timestamp")

    panel = build_panel(bars)
    close = panel["close"].unstack("symbol").sort_index().astype(float)
    volume = panel["volume"].unstack("symbol").sort_index().astype(float)
    if signal_timestamp not in close.index or realization_timestamp not in close.index:
        raise ValueError("paper settlement requires both signal and realization bars")

    cfg = LiquidityExecutionConfig(**manifest.execution_config)
    adv = (close * volume).rolling(cfg.adv_window, min_periods=cfg.adv_window).mean().shift(1)
    available_adv = adv.loc[realization_timestamp].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    current = pd.Series(
        {symbol: float(state.current_weights.get(symbol, 0.0)) for symbol in manifest.symbols},
        index=list(manifest.symbols),
        dtype=float,
    )
    desired = pd.Series(
        {symbol: float(state.pending_target_weights.get(symbol, 0.0)) for symbol in manifest.symbols},
        index=list(manifest.symbols),
        dtype=float,
    )
    delta = desired - current
    desired_notional = delta.abs() * cfg.capital
    max_notional = available_adv.reindex(current.index).fillna(0.0) * cfg.max_participation
    fill_ratio = pd.Series(0.0, index=current.index, dtype=float)
    tradable = desired_notional.gt(0.0) & max_notional.gt(0.0)
    fill_ratio.loc[tradable] = (max_notional.loc[tradable] / desired_notional.loc[tradable]).clip(upper=1.0)
    actual_delta = delta * fill_ratio
    actual_notional = actual_delta.abs() * cfg.capital

    participation = pd.Series(0.0, index=current.index, dtype=float)
    valid_adv = actual_notional.gt(0.0) & available_adv.reindex(current.index).fillna(0.0).gt(0.0)
    participation.loc[valid_adv] = (
        actual_notional.loc[valid_adv] / available_adv.reindex(current.index).loc[valid_adv]
    )
    impact_bps = pd.Series(0.0, index=current.index, dtype=float)
    impact_bps.loc[valid_adv] = cfg.impact_bps_at_one_pct_adv * np.sqrt(participation.loc[valid_adv] / 0.01)
    total_cost_bps = cfg.commission_bps + cfg.spread_bps + impact_bps
    cost_rate = float((actual_delta.abs() * total_cost_bps / 10_000.0).sum())

    realized_weights = current + actual_delta
    previous_close = close.loc[signal_timestamp].reindex(realized_weights.index)
    current_close = close.loc[realization_timestamp].reindex(realized_weights.index)
    asset_returns = current_close / previous_close - 1.0
    gross_return = float((realized_weights * asset_returns.fillna(0.0)).sum())
    net_return = gross_return - cost_rate
    turnover = float(actual_delta.abs().sum())
    total_desired = float(desired_notional.sum())
    total_filled = float(actual_notional.sum())
    fill_rate = 1.0 if total_desired <= 0.0 else float(np.clip(total_filled / total_desired, 0.0, 1.0))
    benchmark_return = float(asset_returns.replace([np.inf, -np.inf], np.nan).dropna().mean()) if asset_returns.notna().any() else 0.0
    signal_count = int(sum(abs(value) > 1e-12 for value in state.pending_target_weights.values()))

    observation = make_paper_observation(
        strategy_id=manifest.strategy_id,
        timestamp=realization_timestamp.isoformat(),
        net_return=net_return,
        benchmark_return=benchmark_return,
        turnover=turnover,
        cost_rate=cost_rate,
        fill_rate=fill_rate,
        signal_count=signal_count,
        notes="automated_frozen_shadow_step",
        research_cycle_id=manifest.research_cycle_id,
        model_artifact_id=manifest.artifact_id,
        signal_timestamp=signal_timestamp.isoformat(),
        signal_snapshot_fingerprint=state.pending_signal_snapshot_fingerprint,
        realization_snapshot_fingerprint=realization_snapshot_fingerprint,
        gross_return=gross_return,
    )
    new_equity = float(state.equity) * (1.0 + net_return)
    new_peak = max(float(state.equity_peak), new_equity)
    return observation, {key: float(value) for key, value in realized_weights.items()}, new_equity, new_peak


def run_shadow_step(
    bars: pd.DataFrame,
    *,
    artifact_dir: str | Path,
    state_path: str | Path,
    ledger_path: str | Path,
    snapshot_fingerprint: str,
    data_age_seconds: float = 0.0,
    kill_switch: bool = False,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> ShadowStepResult:
    """Advance one frozen shadow strategy by one genuinely new market timestamp."""

    snapshot_id = str(snapshot_fingerprint).strip()
    if not snapshot_id:
        raise ValueError("snapshot_fingerprint is required")
    if data_age_seconds < 0.0 or not math.isfinite(float(data_age_seconds)):
        raise ValueError("data_age_seconds must be finite and non-negative")

    manifest = _load_manifest(artifact_dir)
    model = _load_verified_model(artifact_dir, manifest)
    state_target = Path(state_path)
    state = _load_state(state_target, manifest)
    ledger = PaperTradingLedger(ledger_path)

    latest_timestamp, predictions = _latest_predictions(
        bars,
        manifest,
        model,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_feature_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )
    if state.last_processed_timestamp is not None:
        processed_timestamp = pd.Timestamp(state.last_processed_timestamp)
        if latest_timestamp < processed_timestamp:
            raise ValueError("shadow step cannot move market timestamp backwards")
        if latest_timestamp == processed_timestamp:
            if state.pending_signal_snapshot_fingerprint != snapshot_id:
                raise ValueError("shadow snapshot fingerprint changed for already processed market timestamp")
            if state.pending_signal_timestamp != state.last_processed_timestamp:
                raise ValueError("paper runner state is inconsistent for idempotent replay")
            return ShadowStepResult(
                strategy_id=manifest.strategy_id,
                artifact_id=manifest.artifact_id,
                processed_timestamp=state.last_processed_timestamp,
                pending_signal_timestamp=state.pending_signal_timestamp,
                target_weights=dict(state.pending_target_weights),
                observation=None,
                idempotent_replay=True,
                broker_execution_available=False,
            )

    observation: PaperObservation | None = None
    current_weights = dict(state.current_weights)
    equity = float(state.equity)
    equity_peak = float(state.equity_peak)
    if state.pending_signal_timestamp is not None:
        observation, current_weights, equity, equity_peak = _settle_pending_session(
            bars,
            manifest,
            state,
            realization_timestamp=latest_timestamp,
            realization_snapshot_fingerprint=snapshot_id,
        )
        observation = _append_or_verify_observation(ledger, observation)

    signal_weights = _signal_weights(predictions, manifest.top_fraction, manifest.weighting)
    target_row = (
        signal_weights.loc[latest_timestamp]
        if latest_timestamp in signal_weights.index
        else pd.Series(0.0, index=list(manifest.symbols), dtype=float)
    )
    raw_targets = {symbol: float(target_row.get(symbol, 0.0)) for symbol in manifest.symbols}
    risk_state = _paper_risk_state(
        ledger,
        manifest.strategy_id,
        data_age_seconds=float(data_age_seconds),
        kill_switch=bool(kill_switch),
    )
    approved_targets = _risk_adjust_targets(raw_targets, manifest, risk_state)

    next_state = PaperRunnerState(
        strategy_id=manifest.strategy_id,
        artifact_id=manifest.artifact_id,
        last_processed_timestamp=latest_timestamp.isoformat(),
        current_weights=current_weights,
        pending_signal_timestamp=latest_timestamp.isoformat(),
        pending_target_weights=approved_targets,
        pending_signal_snapshot_fingerprint=snapshot_id,
        equity=equity,
        equity_peak=equity_peak,
    )
    _save_state(state_target, next_state)
    return ShadowStepResult(
        strategy_id=manifest.strategy_id,
        artifact_id=manifest.artifact_id,
        processed_timestamp=latest_timestamp.isoformat(),
        pending_signal_timestamp=latest_timestamp.isoformat(),
        target_weights=approved_targets,
        observation=observation,
        idempotent_replay=False,
        broker_execution_available=False,
    )
