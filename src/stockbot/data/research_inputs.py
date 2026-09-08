from __future__ import annotations

import json
from pathlib import Path

from stockbot.data.point_in_time_features import PointInTimeFeatureStore
from stockbot.data.universe import PointInTimeUniverse


RESEARCH_INPUT_SCHEMA_VERSION = 1


def _atomic_write(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _write_artifact(
    path: str | Path,
    *,
    artifact_type: str,
    fingerprint: str,
    payload: dict[str, object],
) -> None:
    _atomic_write(
        path,
        {
            "schema_version": RESEARCH_INPUT_SCHEMA_VERSION,
            "artifact_type": artifact_type,
            "fingerprint": fingerprint,
            "payload": payload,
        },
    )


def _load_artifact(path: str | Path, expected_type: str) -> tuple[str, dict[str, object]]:
    target = Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("invalid research input artifact")
    if int(raw.get("schema_version", -1)) != RESEARCH_INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported research input artifact schema version")
    if raw.get("artifact_type") != expected_type:
        raise ValueError(f"expected {expected_type} research input artifact")
    fingerprint = raw.get("fingerprint")
    payload = raw.get("payload")
    if not isinstance(fingerprint, str) or not isinstance(payload, dict):
        raise ValueError("invalid research input artifact payload")
    return fingerprint, payload


def write_point_in_time_feature_store(
    path: str | Path,
    store: PointInTimeFeatureStore,
) -> None:
    _write_artifact(
        path,
        artifact_type="point_in_time_feature_store",
        fingerprint=store.fingerprint,
        payload=store.to_payload(),
    )


def load_point_in_time_feature_store(path: str | Path) -> PointInTimeFeatureStore:
    expected_fingerprint, payload = _load_artifact(path, "point_in_time_feature_store")
    store = PointInTimeFeatureStore.from_payload(payload)
    if store.fingerprint != expected_fingerprint:
        raise ValueError("point-in-time feature store fingerprint mismatch")
    return store


def write_point_in_time_universe(
    path: str | Path,
    universe: PointInTimeUniverse,
) -> None:
    _write_artifact(
        path,
        artifact_type="point_in_time_universe",
        fingerprint=universe.fingerprint,
        payload=universe.to_payload(),
    )


def load_point_in_time_universe(path: str | Path) -> PointInTimeUniverse:
    expected_fingerprint, payload = _load_artifact(path, "point_in_time_universe")
    universe = PointInTimeUniverse.from_payload(payload)
    if universe.fingerprint != expected_fingerprint:
        raise ValueError("point-in-time universe fingerprint mismatch")
    return universe
