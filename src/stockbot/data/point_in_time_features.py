from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
            ["feature_name", "symbol", "observation_time", "available_time"],
            na_position="first",
            kind="mergesort",
        ).reset_index(drop=True)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._frame["feature_name"].unique()))

    def _value_asof(
        self,
        feature_name: str,
        timestamp: pd.Timestamp,
        symbol: str,
        *,
        max_age_days: int | None,
    ) -> float:
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

    def materialize(
        self,
        index: pd.MultiIndex,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
        max_age_days: int | None = None,
    ) -> pd.DataFrame:
        """Materialize exactly what was knowable at each timestamp/symbol row."""

        if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
            raise ValueError("feature materialization requires a two-level MultiIndex")
        names = tuple(index.names)
        if names == ("symbol", "timestamp"):
            target = index.reorder_levels([1, 0])
        elif names == ("timestamp", "symbol"):
            target = index
        else:
            raise ValueError("index levels must be timestamp/symbol or symbol/timestamp")
        timestamps = pd.to_datetime(target.get_level_values("timestamp"), utc=True)
        symbols = target.get_level_values("symbol").astype(str).str.upper()
        normalized = pd.MultiIndex.from_arrays([timestamps, symbols], names=["timestamp", "symbol"])

        requested = self.feature_names if feature_names is None else tuple(str(value).strip() for value in feature_names)
        if not requested:
            raise ValueError("at least one auxiliary feature is required")
        missing = sorted(set(requested).difference(self.feature_names))
        if missing:
            raise ValueError(f"unknown auxiliary features: {missing}")

        output = pd.DataFrame(index=normalized, columns=requested, dtype=float)
        for timestamp, symbol in normalized:
            for feature_name in requested:
                output.loc[(timestamp, symbol), feature_name] = self._value_asof(
                    feature_name,
                    pd.Timestamp(timestamp),
                    str(symbol),
                    max_age_days=max_age_days,
                )
        return output

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
