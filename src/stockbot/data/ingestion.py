from __future__ import annotations

from pathlib import Path

import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.universe import (
    PointInTimeUniverse,
    UniverseManifest,
    UniverseMembership,
)


FEATURE_REQUIRED_COLUMNS = (
    "feature_name",
    "value",
    "observation_time",
    "available_time",
    "source",
    "kind",
)
FEATURE_OPTIONAL_COLUMNS = ("symbol", "revision_id")
UNIVERSE_REQUIRED_COLUMNS = ("symbol", "effective_from")
UNIVERSE_OPTIONAL_COLUMNS = (
    "effective_to",
    "exchange",
    "sector",
    "industry",
    "delisted",
)


def _validate_columns(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    label: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    allowed = set(required).union(optional)
    unknown = sorted(set(frame.columns).difference(allowed))
    if unknown:
        raise ValueError(f"{label} contains unsupported columns: {unknown}")
    if frame.empty:
        raise ValueError(f"{label} cannot be empty")


def _optional_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_bool(value, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"{label} must be a boolean")


def point_in_time_feature_store_from_frame(
    frame: pd.DataFrame,
    manifest: PointInTimeFeatureManifest,
) -> PointInTimeFeatureStore:
    """Convert a strict provider-normalized frame into the safe auxiliary store.

    `available_time` is mandatory. Importers must never infer it from observation date,
    fiscal period or filename timestamps because that would create look-ahead risk.
    """

    _validate_columns(
        frame,
        required=FEATURE_REQUIRED_COLUMNS,
        optional=FEATURE_OPTIONAL_COLUMNS,
        label="point-in-time feature frame",
    )
    observations: list[PointInTimeFeatureObservation] = []
    for raw in frame.to_dict(orient="records"):
        available = _optional_text(raw["available_time"])
        observation = _optional_text(raw["observation_time"])
        if available is None:
            raise ValueError("available_time cannot be missing")
        if observation is None:
            raise ValueError("observation_time cannot be missing")
        feature_name = _optional_text(raw["feature_name"])
        source = _optional_text(raw["source"])
        kind = _optional_text(raw["kind"])
        if feature_name is None or source is None or kind is None:
            raise ValueError("feature_name, source and kind cannot be missing")
        observations.append(
            PointInTimeFeatureObservation(
                feature_name=feature_name,
                value=float(raw["value"]),
                observation_time=observation,
                available_time=available,
                source=source,
                kind=AuxiliaryFeatureKind(kind.lower()),
                symbol=_optional_text(raw.get("symbol")),
                revision_id=_optional_text(raw.get("revision_id")),
            )
        )
    return PointInTimeFeatureStore(tuple(observations), manifest)


def point_in_time_universe_from_frame(
    frame: pd.DataFrame,
    manifest: UniverseManifest,
) -> PointInTimeUniverse:
    _validate_columns(
        frame,
        required=UNIVERSE_REQUIRED_COLUMNS,
        optional=UNIVERSE_OPTIONAL_COLUMNS,
        label="point-in-time universe frame",
    )
    memberships: list[UniverseMembership] = []
    for raw in frame.to_dict(orient="records"):
        symbol = _optional_text(raw["symbol"])
        effective_from = _optional_text(raw["effective_from"])
        if symbol is None or effective_from is None:
            raise ValueError("symbol and effective_from cannot be missing")
        memberships.append(
            UniverseMembership(
                symbol=symbol,
                effective_from=effective_from,
                effective_to=_optional_text(raw.get("effective_to")),
                exchange=_optional_text(raw.get("exchange")),
                sector=_optional_text(raw.get("sector")),
                industry=_optional_text(raw.get("industry")),
                delisted=_parse_bool(raw.get("delisted", False), label="delisted"),
            )
        )
    return PointInTimeUniverse(tuple(memberships), manifest)


def load_point_in_time_feature_csv(
    path: str | Path,
    manifest: PointInTimeFeatureManifest,
) -> PointInTimeFeatureStore:
    return point_in_time_feature_store_from_frame(pd.read_csv(path), manifest)


def load_point_in_time_universe_csv(
    path: str | Path,
    manifest: UniverseManifest,
) -> PointInTimeUniverse:
    return point_in_time_universe_from_frame(pd.read_csv(path), manifest)
