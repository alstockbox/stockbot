from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stockbot.data.snapshots import SnapshotStore
from stockbot.paper.ledger import PaperObservation
from stockbot.paper.runner import load_frozen_shadow_artifact


@dataclass(frozen=True)
class PaperProvenanceVerificationReport:
    strategy_id: str
    model_artifact_id: str
    research_cycle_id: str
    snapshot_fingerprints: tuple[str, ...]
    snapshots_verified: int
    verified: bool
    broker_execution_available: bool = False
    artifact_verified: bool = False
    model_hash_verified: bool = False
    artifact_strategy_match: bool = False
    artifact_cycle_match: bool = False
    signal_snapshots_verified: int = 0
    realization_snapshots_verified: int = 0
    snapshot_timeline_verified: bool = False
    observations_verified: int = 0
    reasons: tuple[str, ...] = ()


def _required_single_value(
    observations: list[PaperObservation] | tuple[PaperObservation, ...],
    *,
    attribute: str,
    label: str,
) -> str:
    values = []
    for row in observations:
        raw = getattr(row, attribute, None)
        value = "" if raw is None else str(raw).strip()
        if not value:
            raise ValueError(f"paper external provenance requires {label} on every observation")
        values.append(value)
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"paper external provenance contains mixed {label}")
    return values[0]


def _required_text(row: PaperObservation, attribute: str, label: str) -> str:
    raw = getattr(row, attribute, None)
    value = "" if raw is None else str(raw).strip()
    if not value:
        raise ValueError(f"paper external provenance requires {label} on every observation")
    return value


def _utc_timestamp(value: str, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:  # pandas raises several timestamp-specific exception classes.
        raise ValueError(f"{label} must be a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _normalized_symbols(values) -> tuple[str, ...]:
    return tuple(sorted(str(value).strip().upper() for value in values if str(value).strip()))


def _verify_snapshot(
    store: SnapshotStore,
    fingerprint: str,
    *,
    expected_timestamp: str,
    expected_symbols: tuple[str, ...],
    artifact_id: str,
    research_cycle_id: str,
):
    snapshot = store.find_verified_by_fingerprint(fingerprint)
    if snapshot is None:
        raise ValueError(f"paper snapshot fingerprint is not present in the verified snapshot store: {fingerprint}")
    manifest = snapshot.manifest
    if str(manifest.dataset_fingerprint) != fingerprint:
        raise ValueError("paper snapshot fingerprint does not match verified snapshot manifest")
    if _normalized_symbols(manifest.symbols) != expected_symbols:
        raise ValueError("paper snapshot universe does not match frozen artifact universe")

    last_observation = getattr(manifest, "last_observation", None)
    if not isinstance(last_observation, dict) or not last_observation:
        raise ValueError("verified snapshot is missing last-observation timestamp evidence")
    normalized_last_observation = {
        str(symbol).strip().upper(): _utc_timestamp(
            str(value),
            label="snapshot last-observation timestamp",
        )
        for symbol, value in last_observation.items()
        if str(symbol).strip()
    }
    if tuple(sorted(normalized_last_observation)) != expected_symbols:
        raise ValueError("verified snapshot last-observation universe does not match frozen artifact universe")

    expected = _utc_timestamp(expected_timestamp, label="paper provenance timestamp")
    if any(timestamp != expected for timestamp in normalized_last_observation.values()):
        raise ValueError(
            "paper snapshot timestamp does not match the observation provenance timestamp"
        )

    provenance = getattr(manifest, "provenance", {}) or {}
    if not isinstance(provenance, dict):
        raise ValueError("verified snapshot provenance must be an object")
    bound_artifact = provenance.get("shadow_artifact_id")
    if bound_artifact is not None and str(bound_artifact).strip() != artifact_id:
        raise ValueError("paper snapshot provenance is bound to a different frozen artifact")
    bound_cycle = provenance.get("research_cycle_id")
    if bound_cycle is not None and str(bound_cycle).strip() != research_cycle_id:
        raise ValueError("paper snapshot provenance is bound to a different research cycle")
    return snapshot


def _verify_forward_chain_continuity(
    observations: list[PaperObservation] | tuple[PaperObservation, ...],
) -> tuple[PaperObservation, ...]:
    ordered = tuple(
        sorted(
            observations,
            key=lambda row: _utc_timestamp(
                row.timestamp,
                label="paper realization timestamp",
            ),
        )
    )
    for previous, current in zip(ordered, ordered[1:]):
        previous_realization_time = _utc_timestamp(
            previous.timestamp,
            label="paper realization timestamp",
        )
        current_signal_time = _utc_timestamp(
            _required_text(current, "signal_timestamp", "signal timestamp"),
            label="paper signal timestamp",
        )
        previous_realization_fingerprint = _required_text(
            previous,
            "realization_snapshot_fingerprint",
            "realization snapshot fingerprint",
        )
        current_signal_fingerprint = _required_text(
            current,
            "signal_snapshot_fingerprint",
            "signal snapshot fingerprint",
        )
        if (
            current_signal_time != previous_realization_time
            or current_signal_fingerprint != previous_realization_fingerprint
        ):
            raise ValueError(
                "paper forward provenance continuity is broken between consecutive observations"
            )
    return ordered


def verify_frozen_paper_provenance(
    observations: list[PaperObservation] | tuple[PaperObservation, ...],
    *,
    artifact_dir: str | Path,
    snapshot_root: str | Path,
) -> PaperProvenanceVerificationReport:
    """Cryptographically bind paper observations to one frozen artifact and snapshots.

    The frozen artifact is loaded through the normal manifest-identity and model-SHA256
    verifier. Every signal and realization fingerprint must resolve to an immutable
    snapshot that passes the normal snapshot fingerprint checks, has the exact frozen
    symbol universe, and ends at the timestamp recorded by the paper observation.

    Consecutive observations must also form one continuous frozen-forward state chain:
    the previous realization timestamp/fingerprint is exactly the next signal
    timestamp/fingerprint. This prevents individually valid paper fragments from being
    stitched together into a synthetic forward track.

    Snapshot provenance may be generic (for example an explicitly selected pre-existing
    verified snapshot). When it explicitly names a shadow artifact or research cycle,
    that binding must agree with the frozen artifact or verification fails closed.
    """

    if not observations:
        raise ValueError("paper external provenance requires observations")
    strategy_ids = {str(row.strategy_id).strip() for row in observations if str(row.strategy_id).strip()}
    if len(strategy_ids) != 1 or len(strategy_ids) != len({row.strategy_id for row in observations}):
        raise ValueError("paper external provenance requires exactly one strategy_id")
    strategy_id = next(iter(strategy_ids))
    artifact_id = _required_single_value(
        observations,
        attribute="model_artifact_id",
        label="model artifact",
    )
    research_cycle_id = _required_single_value(
        observations,
        attribute="research_cycle_id",
        label="research cycle",
    )
    ordered = _verify_forward_chain_continuity(observations)

    artifact = load_frozen_shadow_artifact(artifact_dir)
    if str(artifact.strategy_id) != strategy_id:
        raise ValueError("paper strategy does not match frozen artifact strategy")
    if str(artifact.artifact_id) != artifact_id:
        raise ValueError("paper model artifact does not match verified frozen artifact")
    if str(artifact.research_cycle_id) != research_cycle_id:
        raise ValueError("paper research cycle does not match verified frozen artifact")
    expected_symbols = _normalized_symbols(artifact.symbols)
    if not expected_symbols:
        raise ValueError("verified frozen artifact universe is empty")

    store = SnapshotStore(snapshot_root)
    verified_snapshots: dict[str, object] = {}
    verified_signal_fingerprints: set[str] = set()
    verified_realization_fingerprints: set[str] = set()
    for row in ordered:
        signal_timestamp = _required_text(row, "signal_timestamp", "signal timestamp")
        signal_fingerprint = _required_text(
            row,
            "signal_snapshot_fingerprint",
            "signal snapshot fingerprint",
        )
        realization_fingerprint = _required_text(
            row,
            "realization_snapshot_fingerprint",
            "realization snapshot fingerprint",
        )

        signal_snapshot = _verify_snapshot(
            store,
            signal_fingerprint,
            expected_timestamp=signal_timestamp,
            expected_symbols=expected_symbols,
            artifact_id=artifact_id,
            research_cycle_id=research_cycle_id,
        )
        verified_snapshots[signal_fingerprint] = signal_snapshot
        verified_signal_fingerprints.add(signal_fingerprint)
        realization_snapshot = _verify_snapshot(
            store,
            realization_fingerprint,
            expected_timestamp=row.timestamp,
            expected_symbols=expected_symbols,
            artifact_id=artifact_id,
            research_cycle_id=research_cycle_id,
        )
        verified_snapshots[realization_fingerprint] = realization_snapshot
        verified_realization_fingerprints.add(realization_fingerprint)

    fingerprints = tuple(sorted(verified_snapshots))
    return PaperProvenanceVerificationReport(
        strategy_id=strategy_id,
        model_artifact_id=artifact_id,
        research_cycle_id=research_cycle_id,
        snapshot_fingerprints=fingerprints,
        snapshots_verified=len(fingerprints),
        verified=True,
        broker_execution_available=False,
        artifact_verified=True,
        model_hash_verified=True,
        artifact_strategy_match=True,
        artifact_cycle_match=True,
        signal_snapshots_verified=len(verified_signal_fingerprints),
        realization_snapshots_verified=len(verified_realization_fingerprints),
        snapshot_timeline_verified=True,
        observations_verified=len(ordered),
        reasons=(),
    )
