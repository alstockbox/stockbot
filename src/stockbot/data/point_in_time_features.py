from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json

import numpy as np
import pandas as pd


class AuxiliaryFeatureKind(str, Enum):
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"
    NEWS = "news"
    SENTIMENT = "sentiment"
    SHORT_INTEREST = "short_interest"
    INSIDER = "insider"
    ALTERNATIVE = "alternative"


@dataclass(frozen=True)
class PointInTimeFeatureObservation:
    feature_name: str
    value: float
    observation_time: str
    available_time: str
    source: str
    kind: AuxiliaryFeatureKind
    symbol: str | None = None
    revision_id: str | None = None


@dataclass(frozen=True)
class PointInTimeFeatureManifest:
    source: str
    point_in_time: bool
    revision_aware: bool
    available_time_semantics: str = "first_usable_by_strategy"


@dataclass(frozen=True)
class AuxiliaryFeatureCoverageReport:
    requested_features: tuple[str, ...]
    total_cells: int
    covered_cells: int
    coverage: float
    feature_coverage: dict[str, float]
    point_in_time: bool
    revision_aware: bool
    research_grade_auxiliary: bool
    reasons: tuple[str, ...]


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


class PointInTimeFeatureStore:
    """Provider-neutral point-in-time store for non-price research features.

    `observation_time` describes the economic period/event being measured while
    `available_time` is the earliest timestamp at which the strategy could have used
    that exact value. Materialization always filters on `available_time <= asof`, so
    delayed filings and later revisions cannot leak backwards into historical rows.
    Symbol=None denotes a global feature (for example policy rate or CPI surprise)
    which is broadcast to every symbol at a given timestamp.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        observations: list[PointInTimeFeatureObservation] | tuple[PointInTimeFeatureObservation, ...],
        manifest: PointInTimeFeatureManifest,
    ) -> None:
        if not observations:
            raise ValueError("point-in-time feature store requires observations")
        self.observations = tuple(observations)
        self.manifest = manifest
        self._frame = self._build_frame()
        self._timeline_cache: dict[tuple[str, str | None], pd.DataFrame] = {}

    def _build_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        seen: set[tuple[object, ...]] = set()
        for row in self.observations:
            feature_name = row.feature_name.strip()
            source = row.source.strip()
            if not feature_name:
                raise ValueError("feature_name cannot be empty")
            if not source:
                raise ValueError("feature source cannot be empty")
            value = float(row.value)
            if not np.isfinite(value):
                raise ValueError("point-in-time feature values must be finite")
            observation_time = _utc(row.observation_time)
            available_time = _utc(row.available_time)
            if available_time < observation_time:
                raise ValueError("available_time cannot precede observation_time")
            symbol = None if row.symbol is None else row.symbol.strip().upper()
            if row.symbol is not None and not symbol:
                raise ValueError("feature symbol cannot be blank")
            revision_id = None if row.revision_id is None else str(row.revision_id)
            key = (feature_name, symbol, observation_time, available_time, revision_id)
            if key in seen:
                raise ValueError("duplicate point-in-time feature observation")
            seen.add(key)
            rows.append(
                {
                    "feature_name": feature_name,
                    "value": value,
                    "observation_time": observation_time,
                    "available_time": available_time,
                    "source": source,
                    "kind": row.kind.value,
                    "symbol": symbol,
                    "revision_id": revision_id,
                }
            )
        frame = pd.DataFrame(rows)
        return frame.sort_values(
            ["feature_name", "symbol", "observation_time", "available_time", "revision_id"],
            na_position="first",
            kind="mergesort",
        ).reset_index(drop=True)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._frame["feature_name"].unique()))

    def to_payload(self) -> dict[str, object]:
        """Return a canonical JSON-safe payload used for persistence and identity."""

        manifest = asdict(self.manifest)
        observations: list[dict[str, object]] = []
        for row in self._frame.itertuples(index=False):
            observations.append(
                {
                    "feature_name": str(row.feature_name),
                    "value_hex": float(row.value).hex(),
                    "observation_time": pd.Timestamp(row.observation_time).isoformat(),
                    "available_time": pd.Timestamp(row.available_time).isoformat(),
                    "source": str(row.source),
                    "kind": str(row.kind),
                    "symbol": None if pd.isna(row.symbol) else str(row.symbol),
                    "revision_id": None if pd.isna(row.revision_id) else str(row.revision_id),
                }
            )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "manifest": manifest,
            "observations": observations,
        }

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(
            self.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "PointInTimeFeatureStore":
        if int(payload.get("schema_version", -1)) != cls.SCHEMA_VERSION:
            raise ValueError("unsupported point-in-time feature store schema version")
        raw_manifest = payload.get("manifest")
        raw_observations = payload.get("observations")
        if not isinstance(raw_manifest, dict) or not isinstance(raw_observations, list):
            raise ValueError("invalid point-in-time feature store payload")
        manifest = PointInTimeFeatureManifest(
            source=str(raw_manifest["source"]),
            point_in_time=bool(raw_manifest["point_in_time"]),
            revision_aware=bool(raw_manifest["revision_aware"]),
            available_time_semantics=str(
                raw_manifest.get("available_time_semantics", "first_usable_by_strategy")
            ),
        )
        observations: list[PointInTimeFeatureObservation] = []
        for raw in raw_observations:
            if not isinstance(raw, dict):
                raise ValueError("invalid point-in-time feature observation payload")
            observations.append(
                PointInTimeFeatureObservation(
                    feature_name=str(raw["feature_name"]),
                    value=float.fromhex(str(raw["value_hex"])),
                    observation_time=str(raw["observation_time"]),
                    available_time=str(raw["available_time"]),
                    source=str(raw["source"]),
                    kind=AuxiliaryFeatureKind(str(raw["kind"])),
                    symbol=None if raw.get("symbol") is None else str(raw["symbol"]),
                    revision_id=(
                        None if raw.get("revision_id") is None else str(raw["revision_id"])
                    ),
                )
            )
        return cls(tuple(observations), manifest)

    @staticmethod
    def _revision_key(value: object) -> tuple[int, str]:
        if pd.isna(value):
            return (0, "")
        return (1, str(value))

    def _state_timeline(self, feature_name: str, symbol: str | None) -> pd.DataFrame:
        """Return only the selected state visible after each source publication time.

        The legacy materializer selected the lexicographically latest
        `(observation_time, available_time, revision_id)` among observations known at
        each as-of timestamp. Building that state once per feature/scope makes repeated
        historical materialization a binary-search problem instead of repeated DataFrame
        filtering while preserving identical revision semantics.
        """

        cache_key = (feature_name, symbol)
        cached = self._timeline_cache.get(cache_key)
        if cached is not None:
            return cached

        feature_rows = self._frame[self._frame["feature_name"] == feature_name]
        if symbol is None:
            rows = feature_rows[feature_rows["symbol"].isna()].copy()
        else:
            rows = feature_rows[feature_rows["symbol"] == symbol].copy()
        if rows.empty:
            timeline = pd.DataFrame(
                columns=["available_time", "observation_time", "value"],
            )
            self._timeline_cache[cache_key] = timeline
            return timeline

        rows = rows.sort_values(
            ["available_time", "observation_time", "revision_id"],
            na_position="first",
            kind="mergesort",
        )
        best_key: tuple[int, int, tuple[int, str]] | None = None
        best_row = None
        states: list[dict[str, object]] = []

        for available_time, group in rows.groupby("available_time", sort=True):
            for row in group.itertuples(index=False):
                candidate_key = (
                    int(pd.Timestamp(row.observation_time).value),
                    int(pd.Timestamp(row.available_time).value),
                    self._revision_key(row.revision_id),
                )
                if best_key is None or candidate_key >= best_key:
                    best_key = candidate_key
                    best_row = row
            if best_row is not None:
                states.append(
                    {
                        "available_time": pd.Timestamp(available_time),
                        "observation_time": pd.Timestamp(best_row.observation_time),
                        "value": float(best_row.value),
                    }
                )

        timeline = pd.DataFrame(states).sort_values("available_time", kind="mergesort").reset_index(drop=True)
        self._timeline_cache[cache_key] = timeline
        return timeline

    @staticmethod
    def _timeline_values(
        timeline: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        *,
        max_age_days: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.full(len(timestamps), np.nan, dtype=float)
        known = np.zeros(len(timestamps), dtype=bool)
        if timeline.empty or len(timestamps) == 0:
            return values, known

        available_ns = pd.DatetimeIndex(timeline["available_time"]).asi8
        target_ns = timestamps.asi8
        positions = np.searchsorted(available_ns, target_ns, side="right") - 1
        known = positions >= 0
        if not known.any():
            return values, known

        known_positions = positions[known]
        selected_values = timeline["value"].to_numpy(dtype=float)[known_positions]
        if max_age_days is not None:
            observation_ns = pd.DatetimeIndex(timeline["observation_time"]).asi8[known_positions]
            ages_days = (target_ns[known] - observation_ns) / (86400.0 * 1_000_000_000.0)
            fresh = ages_days <= float(max_age_days)
            known_indices = np.flatnonzero(known)
            values[known_indices[fresh]] = selected_values[fresh]
            final_known = np.zeros(len(timestamps), dtype=bool)
            final_known[known_indices[fresh]] = True
            return values, final_known

        values[known] = selected_values
        return values, known

    def _value_asof(
        self,
        feature_name: str,
        timestamp: pd.Timestamp,
        symbol: str,
        *,
        max_age_days: int | None,
    ) -> float:
        """Reference single-cell implementation retained for parity tests/debugging."""

        subset = self._frame[
            (self._frame["feature_name"] == feature_name)
            & (self._frame["available_time"] <= timestamp)
            & (self._frame["observation_time"] <= timestamp)
            & (
                self._frame["symbol"].isna()
                | (self._frame["symbol"] == symbol)
            )
        ]
        if subset.empty:
            return float("nan")
        symbol_specific = subset[subset["symbol"] == symbol]
        if not symbol_specific.empty:
            subset = symbol_specific
        latest = subset.sort_values(
            ["observation_time", "available_time"], kind="mergesort"
        ).iloc[-1]
        if max_age_days is not None:
            if max_age_days < 0:
                raise ValueError("max_age_days cannot be negative")
            age_days = (timestamp - pd.Timestamp(latest["observation_time"])).total_seconds() / 86400.0
            if age_days > max_age_days:
                return float("nan")
        return float(latest["value"])

    @staticmethod
    def _normalize_index(index: pd.MultiIndex) -> tuple[pd.MultiIndex, pd.DatetimeIndex, np.ndarray]:
        if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
            raise ValueError("feature materialization requires a two-level MultiIndex")
        names = tuple(index.names)
        if names == ("symbol", "timestamp"):
            target = index.reorder_levels([1, 0])
        elif names == ("timestamp", "symbol"):
            target = index
        else:
            raise ValueError("index levels must be timestamp/symbol or symbol/timestamp")
        timestamps = pd.DatetimeIndex(pd.to_datetime(target.get_level_values("timestamp"), utc=True))
        symbols = np.asarray(
            [str(value).upper() for value in target.get_level_values("symbol")],
            dtype=object,
        )
        normalized = pd.MultiIndex.from_arrays(
            [timestamps, symbols],
            names=["timestamp", "symbol"],
        )
        return normalized, timestamps, symbols

    def materialize(
        self,
        index: pd.MultiIndex,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
        max_age_days: int | None = None,
    ) -> pd.DataFrame:
        """Materialize exactly what was knowable at each timestamp/symbol row.

        State timelines are cached per feature/global-or-symbol scope and rows use
        binary search against `available_time`. This preserves the original point-in-time
        selection semantics while scaling to much larger universes and date ranges.
        """

        if max_age_days is not None and max_age_days < 0:
            raise ValueError("max_age_days cannot be negative")
        normalized, timestamps, symbols = self._normalize_index(index)

        requested = self.feature_names if feature_names is None else tuple(str(value).strip() for value in feature_names)
        if not requested:
            raise ValueError("at least one auxiliary feature is required")
        missing = sorted(set(requested).difference(self.feature_names))
        if missing:
            raise ValueError(f"unknown auxiliary features: {missing}")

        unique_requested = tuple(dict.fromkeys(requested))
        materialized: dict[str, np.ndarray] = {}
        unique_symbols = tuple(dict.fromkeys(str(value) for value in symbols))

        for feature_name in unique_requested:
            global_values, _ = self._timeline_values(
                self._state_timeline(feature_name, None),
                timestamps,
                max_age_days=max_age_days,
            )
            values = global_values.copy()

            for symbol in unique_symbols:
                row_mask = symbols == symbol
                if not row_mask.any():
                    continue
                symbol_timestamps = timestamps[row_mask]
                specific_values, specific_known = self._timeline_values(
                    self._state_timeline(feature_name, symbol),
                    symbol_timestamps,
                    max_age_days=max_age_days,
                )
                if specific_known.any():
                    target_rows = np.flatnonzero(row_mask)
                    values[target_rows[specific_known]] = specific_values[specific_known]
            materialized[feature_name] = values

        base = pd.DataFrame(materialized, index=normalized, dtype=float)
        return base.loc[:, list(requested)]

    def coverage_for_index(
        self,
        index: pd.MultiIndex,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
        max_age_days: int | None = None,
        min_coverage: float = 0.80,
    ) -> AuxiliaryFeatureCoverageReport:
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0,1]")
        frame = self.materialize(index, feature_names=feature_names, max_age_days=max_age_days)
        total_cells = int(frame.shape[0] * frame.shape[1])
        covered_cells = int(frame.notna().sum().sum())
        coverage = covered_cells / total_cells if total_cells else 0.0
        feature_coverage = {
            column: float(frame[column].notna().mean()) for column in frame.columns
        }
        reasons: list[str] = []
        if not self.manifest.point_in_time:
            reasons.append("auxiliary_data_not_point_in_time")
        if not self.manifest.revision_aware:
            reasons.append("auxiliary_data_not_revision_aware")
        if coverage < min_coverage:
            reasons.append("insufficient_auxiliary_feature_coverage")
        return AuxiliaryFeatureCoverageReport(
            requested_features=tuple(frame.columns),
            total_cells=total_cells,
            covered_cells=covered_cells,
            coverage=float(coverage),
            feature_coverage=feature_coverage,
            point_in_time=self.manifest.point_in_time,
            revision_aware=self.manifest.revision_aware,
            research_grade_auxiliary=not reasons,
            reasons=tuple(reasons),
        )
