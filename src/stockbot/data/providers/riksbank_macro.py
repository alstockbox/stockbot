from __future__ import annotations

from dataclasses import dataclass
import math
from urllib.parse import urlencode

import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.providers.http import HttpTransport, ProviderError


BASE_URL = "https://api.riksbank.se/monetary_policy_data/v1/forecasts"


@dataclass(frozen=True)
class RiksbankSeries:
    feature_name: str
    series_id: str
    kind: AuxiliaryFeatureKind = AuxiliaryFeatureKind.MACRO


DEFAULT_SERIES = (
    RiksbankSeries("policy_rate", "SEQRATENAYNA"),
    RiksbankSeries("cpif_yoy", "SEMCPIFNAYNA"),
    RiksbankSeries("cpi_yoy", "SEMCPINAYNA"),
    RiksbankSeries("gdp_yoy_ca", "SEQGDPNAYCA"),
    RiksbankSeries("unemployment_rate", "SEQLABUEASA"),
    RiksbankSeries("kix_index", "SEQKIXNAANA"),
)


_DATE_KEYS = (
    "date",
    "period",
    "observation_date",
    "observationDate",
    "forecast_date",
    "forecastDate",
    "time_period",
    "timePeriod",
)
_AVAILABLE_KEYS = (
    "cutoff_date",
    "cutoffDate",
    "cutOffDate",
    "publication_date",
    "publicationDate",
    "published_at",
    "publishedAt",
    "available_time",
    "availableTime",
)
_VALUE_KEYS = ("value", "Value")
_ROUND_KEYS = ("policy_round_name", "policyRoundName", "policy_round", "policyRound")


def _first(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _utc_timestamp(value, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ProviderError(f"Riksbank returned invalid {label}") from exc
    if pd.isna(timestamp):
        raise ProviderError(f"Riksbank returned invalid {label}")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _rows_from_payload(payload) -> tuple[list[dict], dict]:
    if isinstance(payload, list):
        rows = payload
        metadata: dict = {}
    elif isinstance(payload, dict):
        metadata = payload
        rows = None
        for key in ("data", "values", "observations", "results", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            if _first(payload, _VALUE_KEYS) is not None and _first(payload, _DATE_KEYS) is not None:
                rows = [payload]
            else:
                raise ProviderError("Riksbank returned unsupported monetary-policy payload shape")
    else:
        raise ProviderError("Riksbank returned unsupported monetary-policy payload type")

    if not rows:
        raise ProviderError("Riksbank returned no monetary-policy rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ProviderError("Riksbank returned malformed monetary-policy row")
    return list(rows), metadata


class RiksbankMonetaryPolicyProvider:
    """Point-in-time macro adapter for Sveriges Riksbank Monetary Policy Data API.

    The provider deliberately refuses to infer publication time. A row is converted to
    a StockBot auxiliary observation only when a cutoff/publication timestamp is present
    either on the row itself or in response-level metadata. V1 materializes realised
    observations only; future forecast targets are skipped until they have an explicit
    horizon-aware feature schema.
    """

    name = "riksbank-monetary-policy"

    def __init__(self, transport=None) -> None:
        self._transport = transport or HttpTransport()

    def list_policy_rounds(self):
        return self._transport.get_json(
            f"{BASE_URL}/policy_rounds",
            {"Accept": "application/json", "User-Agent": "StockBot/2"},
        )

    def list_series(self):
        return self._transport.get_json(
            f"{BASE_URL}/series_ids",
            {"Accept": "application/json", "User-Agent": "StockBot/2"},
        )

    def fetch_series_payload(self, series_id: str, *, policy_round: str | None = None):
        series = str(series_id or "").strip()
        if not series:
            raise ProviderError("Riksbank series_id is required")
        params = {"series": series}
        if policy_round is not None:
            round_name = str(policy_round).strip()
            if not round_name:
                raise ProviderError("Riksbank policy_round cannot be blank")
            params["policy_round_name"] = round_name
        url = f"{BASE_URL}?{urlencode(params)}"
        return self._transport.get_json(
            url,
            {"Accept": "application/json", "User-Agent": "StockBot/2"},
        )

    def observations_from_payload(
        self,
        payload,
        *,
        feature_name: str,
        series_id: str,
        kind: AuxiliaryFeatureKind = AuxiliaryFeatureKind.MACRO,
        realised_only: bool = True,
    ) -> tuple[PointInTimeFeatureObservation, ...]:
        feature = str(feature_name or "").strip()
        series = str(series_id or "").strip()
        if not feature or not series:
            raise ProviderError("feature_name and series_id are required")

        rows, metadata = _rows_from_payload(payload)
        metadata_available = _first(metadata, _AVAILABLE_KEYS)
        metadata_round = _first(metadata, _ROUND_KEYS)
        output: list[PointInTimeFeatureObservation] = []

        for row in rows:
            raw_value = _first(row, _VALUE_KEYS)
            raw_date = _first(row, _DATE_KEYS)
            raw_available = _first(row, _AVAILABLE_KEYS)
            if raw_available is None:
                raw_available = metadata_available
            if raw_value is None or raw_date is None:
                raise ProviderError("Riksbank monetary-policy row lacks value/date")
            if raw_available is None:
                raise ProviderError("Riksbank monetary-policy row lacks cutoff/publication time")

            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ProviderError("Riksbank monetary-policy value is not numeric") from exc
            if not math.isfinite(value):
                raise ProviderError("Riksbank monetary-policy value is not finite")

            observation_time = _utc_timestamp(raw_date, label="observation date")
            available_time = _utc_timestamp(raw_available, label="cutoff/publication date")
            is_future_target = observation_time > available_time
            if is_future_target and realised_only:
                continue
            if is_future_target:
                raise ProviderError(
                    "future Riksbank forecast rows require an explicit horizon-aware feature schema"
                )

            round_name = _first(row, _ROUND_KEYS) or metadata_round
            revision_parts = [f"series={series}"]
            if round_name is not None:
                revision_parts.append(f"round={round_name}")

            output.append(
                PointInTimeFeatureObservation(
                    feature_name=feature,
                    value=value,
                    observation_time=observation_time.isoformat(),
                    available_time=available_time.isoformat(),
                    source=self.name,
                    kind=kind,
                    symbol=None,
                    revision_id="|".join(revision_parts),
                )
            )

        if not output:
            raise ProviderError("Riksbank payload contained no realised rows usable by the strategy")
        return tuple(output)

    def fetch_feature(
        self,
        series: RiksbankSeries,
        *,
        policy_round: str | None = None,
    ) -> tuple[PointInTimeFeatureObservation, ...]:
        payload = self.fetch_series_payload(series.series_id, policy_round=policy_round)
        return self.observations_from_payload(
            payload,
            feature_name=series.feature_name,
            series_id=series.series_id,
            kind=series.kind,
        )

    def build_store(
        self,
        series: tuple[RiksbankSeries, ...] | list[RiksbankSeries] = DEFAULT_SERIES,
        *,
        policy_round: str | None = None,
    ) -> PointInTimeFeatureStore:
        requested = tuple(series)
        if not requested:
            raise ProviderError("at least one Riksbank series is required")
        observations: list[PointInTimeFeatureObservation] = []
        for item in requested:
            observations.extend(self.fetch_feature(item, policy_round=policy_round))
        return PointInTimeFeatureStore(
            observations,
            PointInTimeFeatureManifest(
                source=self.name,
                point_in_time=True,
                revision_aware=True,
                available_time_semantics="riksbank_cutoff_or_publication_time",
            ),
        )


__all__ = [
    "BASE_URL",
    "DEFAULT_SERIES",
    "RiksbankMonetaryPolicyProvider",
    "RiksbankSeries",
]
