from __future__ import annotations

from dataclasses import dataclass
import math
import re
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


@dataclass(frozen=True)
class RiksbankForecastHorizon:
    suffix: str
    days: int
    tolerance_days: int

    def __post_init__(self) -> None:
        if not self.suffix.strip():
            raise ValueError("forecast horizon suffix cannot be empty")
        if self.days <= 0:
            raise ValueError("forecast horizon days must be positive")
        if self.tolerance_days < 0:
            raise ValueError("forecast horizon tolerance_days cannot be negative")


DEFAULT_SERIES = (
    RiksbankSeries("policy_rate", "SEQRATENAYNA"),
    RiksbankSeries("cpif_yoy", "SEMCPIFNAYNA"),
    RiksbankSeries("cpi_yoy", "SEMCPINAYNA"),
    RiksbankSeries("gdp_yoy_ca", "SEQGDPNAYCA"),
    RiksbankSeries("unemployment_rate", "SEQLABUEASA"),
    RiksbankSeries("kix_index", "SEQKIXNAANA"),
)

DEFAULT_FORECAST_HORIZONS = (
    RiksbankForecastHorizon("3m", 91, 50),
    RiksbankForecastHorizon("6m", 182, 70),
    RiksbankForecastHorizon("12m", 365, 100),
)


_DATE_KEYS = (
    "date",
    "dt",
    "period",
    "observation_date",
    "observationDate",
    "forecast_date",
    "forecastDate",
    "time_period",
    "timePeriod",
)
_AVAILABLE_KEYS = (
    "policy_round_end_dtm",
    "policyRoundEndDtm",
    "publication_date",
    "publicationDate",
    "published_at",
    "publishedAt",
    "available_time",
    "availableTime",
    "forecast_cutoff_date",
    "forecastCutoffDate",
    "cutoff_date",
    "cutoffDate",
    "cutOffDate",
)
_VALUE_KEYS = ("value", "Value")
_ROUND_KEYS = ("policy_round_name", "policyRoundName", "policy_round", "policyRound")
_ROUND_LIST_KEYS = ("policy_rounds", "policyRounds", "data", "values", "results", "items")
_ROUND_PATTERN = re.compile(r"^(?P<year>\d{4}):(?P<number>\d+)$")


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


def _numeric_value(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderError("Riksbank monetary-policy value is not numeric") from exc
    if not math.isfinite(result):
        raise ProviderError("Riksbank monetary-policy value is not finite")
    return result


def _flatten_live_vintages(rows: list[dict]) -> list[dict]:
    nested_flags = ["vintages" in row for row in rows]
    if not any(nested_flags):
        return rows
    if not all(nested_flags):
        raise ProviderError("Riksbank returned mixed monetary-policy payload shapes")

    flattened: list[dict] = []
    for series_row in rows:
        raw_vintages = series_row.get("vintages")
        if isinstance(raw_vintages, dict):
            vintages = [raw_vintages]
        elif isinstance(raw_vintages, list):
            vintages = raw_vintages
        else:
            raise ProviderError("Riksbank returned malformed monetary-policy vintages")
        if not vintages:
            raise ProviderError("Riksbank returned no monetary-policy vintages")

        for vintage in vintages:
            if not isinstance(vintage, dict):
                raise ProviderError("Riksbank returned malformed monetary-policy vintage")
            metadata = vintage.get("metadata")
            observations = vintage.get("observations")
            if not isinstance(metadata, dict):
                raise ProviderError("Riksbank monetary-policy vintage lacks metadata")
            if not isinstance(observations, list) or not observations:
                raise ProviderError("Riksbank monetary-policy vintage lacks observations")

            available = _first(metadata, _AVAILABLE_KEYS)
            round_name = _first(metadata, _ROUND_KEYS)
            if available is None:
                raise ProviderError("Riksbank monetary-policy vintage lacks cutoff/publication time")
            if round_name is None:
                raise ProviderError("Riksbank monetary-policy vintage lacks policy round")

            for observation in observations:
                if not isinstance(observation, dict):
                    raise ProviderError("Riksbank returned malformed monetary-policy observation")
                row = dict(observation)
                row.setdefault("available_time", available)
                row.setdefault("policy_round", round_name)
                flattened.append(row)

    return flattened


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
    return _flatten_live_vintages(list(rows)), metadata


def policy_round_names(payload) -> tuple[str, ...]:
    """Extract and chronologically sort historical policy-round identifiers."""

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in _ROUND_LIST_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise ProviderError("Riksbank returned unsupported policy-round payload shape")
    else:
        raise ProviderError("Riksbank returned unsupported policy-round payload type")

    names: set[str] = set()
    for row in rows:
        if isinstance(row, str):
            name = row.strip()
        elif isinstance(row, dict):
            raw = _first(row, _ROUND_KEYS)
            if raw is None:
                raw = row.get("name")
            name = str(raw or "").strip()
        else:
            raise ProviderError("Riksbank returned malformed policy-round row")
        match = _ROUND_PATTERN.fullmatch(name)
        if match is None:
            raise ProviderError(f"Riksbank returned invalid policy-round identifier: {name!r}")
        names.add(name)

    if not names:
        raise ProviderError("Riksbank returned no policy rounds")

    return tuple(
        sorted(
            names,
            key=lambda name: (
                int(_ROUND_PATTERN.fullmatch(name).group("year")),
                int(_ROUND_PATTERN.fullmatch(name).group("number")),
            ),
        )
    )


class RiksbankMonetaryPolicyProvider:
    """Point-in-time macro adapter for Sveriges Riksbank Monetary Policy Data API.

    Publication timing is never inferred. Realised values are materialized at the
    source cutoff/publication timestamp. Optional forecast features use stable fixed
    horizons and select the future target closest to each horizon within a configured
    tolerance, preserving a deterministic feature schema across policy rounds.
    """

    name = "riksbank-monetary-policy"

    def __init__(self, transport=None) -> None:
        self._transport = transport or HttpTransport()

    def list_policy_rounds(self):
        return self._transport.get_json(
            f"{BASE_URL}/policy_rounds",
            {"Accept": "application/json", "User-Agent": "StockBot/2"},
        )

    def policy_round_names(
        self,
        *,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> tuple[str, ...]:
        if start_year is not None and start_year < 2000:
            raise ProviderError("Riksbank start_year is implausibly early")
        if end_year is not None and end_year < 2000:
            raise ProviderError("Riksbank end_year is implausibly early")
        if start_year is not None and end_year is not None and start_year > end_year:
            raise ProviderError("Riksbank start_year cannot exceed end_year")

        names = policy_round_names(self.list_policy_rounds())
        filtered = tuple(
            name
            for name in names
            if (start_year is None or int(name[:4]) >= start_year)
            and (end_year is None or int(name[:4]) <= end_year)
        )
        if not filtered:
            raise ProviderError("no Riksbank policy rounds matched the requested year range")
        return filtered

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
            if _ROUND_PATTERN.fullmatch(round_name) is None and round_name != "latest":
                raise ProviderError("Riksbank policy_round must be YYYY:N or 'latest'")
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

            value = _numeric_value(raw_value)
            observation_time = _utc_timestamp(raw_date, label="observation date")
            available_time = _utc_timestamp(raw_available, label="cutoff/publication date")
            is_future_target = observation_time > available_time
            if is_future_target and realised_only:
                continue
            if is_future_target:
                raise ProviderError(
                    "future Riksbank forecast rows require forecast_observations_from_payload"
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

    def forecast_observations_from_payload(
        self,
        payload,
        *,
        feature_name: str,
        series_id: str,
        kind: AuxiliaryFeatureKind = AuxiliaryFeatureKind.MACRO,
        horizons: tuple[RiksbankForecastHorizon, ...] | list[RiksbankForecastHorizon] = DEFAULT_FORECAST_HORIZONS,
    ) -> tuple[PointInTimeFeatureObservation, ...]:
        """Convert future targets into stable fixed-horizon vintage features."""

        feature = str(feature_name or "").strip()
        series = str(series_id or "").strip()
        selected_horizons = tuple(horizons)
        if not feature or not series:
            raise ProviderError("feature_name and series_id are required")
        if not selected_horizons:
            raise ProviderError("at least one Riksbank forecast horizon is required")
        if len({item.suffix for item in selected_horizons}) != len(selected_horizons):
            raise ProviderError("Riksbank forecast horizon suffixes must be unique")

        rows, metadata = _rows_from_payload(payload)
        metadata_available = _first(metadata, _AVAILABLE_KEYS)
        metadata_round = _first(metadata, _ROUND_KEYS)
        grouped: dict[pd.Timestamp, list[tuple[pd.Timestamp, float, str | None]]] = {}

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

            target_time = _utc_timestamp(raw_date, label="forecast target date")
            available_time = _utc_timestamp(raw_available, label="cutoff/publication date")
            if target_time <= available_time:
                continue
            round_name = _first(row, _ROUND_KEYS) or metadata_round
            grouped.setdefault(available_time, []).append(
                (target_time, _numeric_value(raw_value), None if round_name is None else str(round_name))
            )

        output: list[PointInTimeFeatureObservation] = []
        for available_time in sorted(grouped):
            candidates = grouped[available_time]
            for horizon in selected_horizons:
                anchor = available_time + pd.Timedelta(days=horizon.days)
                target_time, value, round_name = min(
                    candidates,
                    key=lambda item: (abs((item[0] - anchor).total_seconds()), item[0]),
                )
                distance_days = abs((target_time - anchor).total_seconds()) / 86400.0
                if distance_days > horizon.tolerance_days:
                    continue
                revision_parts = [
                    f"series={series}",
                    f"target={target_time.isoformat()}",
                    f"horizon={horizon.suffix}",
                ]
                if round_name is not None:
                    revision_parts.append(f"round={round_name}")
                output.append(
                    PointInTimeFeatureObservation(
                        feature_name=f"{feature}__forecast_{horizon.suffix}",
                        value=value,
                        observation_time=available_time.isoformat(),
                        available_time=available_time.isoformat(),
                        source=self.name,
                        kind=kind,
                        symbol=None,
                        revision_id="|".join(revision_parts),
                    )
                )
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

    @staticmethod
    def _store(observations: list[PointInTimeFeatureObservation]) -> PointInTimeFeatureStore:
        return PointInTimeFeatureStore(
            observations,
            PointInTimeFeatureManifest(
                source=RiksbankMonetaryPolicyProvider.name,
                point_in_time=True,
                revision_aware=True,
                available_time_semantics="riksbank_cutoff_or_publication_time",
            ),
        )

    def build_store(
        self,
        series: tuple[RiksbankSeries, ...] | list[RiksbankSeries] = DEFAULT_SERIES,
        *,
        policy_round: str | None = None,
        include_forecasts: bool = False,
        forecast_horizons: tuple[RiksbankForecastHorizon, ...] | list[RiksbankForecastHorizon] = DEFAULT_FORECAST_HORIZONS,
    ) -> PointInTimeFeatureStore:
        requested = tuple(series)
        if not requested:
            raise ProviderError("at least one Riksbank series is required")
        observations: list[PointInTimeFeatureObservation] = []
        for item in requested:
            payload = self.fetch_series_payload(item.series_id, policy_round=policy_round)
            observations.extend(
                self.observations_from_payload(
                    payload,
                    feature_name=item.feature_name,
                    series_id=item.series_id,
                    kind=item.kind,
                )
            )
            if include_forecasts:
                observations.extend(
                    self.forecast_observations_from_payload(
                        payload,
                        feature_name=item.feature_name,
                        series_id=item.series_id,
                        kind=item.kind,
                        horizons=forecast_horizons,
                    )
                )
        return self._store(observations)

    def build_vintage_store(
        self,
        series: tuple[RiksbankSeries, ...] | list[RiksbankSeries] = DEFAULT_SERIES,
        *,
        start_year: int | None = 2020,
        end_year: int | None = None,
        include_forecasts: bool = False,
        forecast_horizons: tuple[RiksbankForecastHorizon, ...] | list[RiksbankForecastHorizon] = DEFAULT_FORECAST_HORIZONS,
    ) -> PointInTimeFeatureStore:
        """Collect requested series for every historical policy-round vintage."""

        requested = tuple(series)
        if not requested:
            raise ProviderError("at least one Riksbank series is required")
        rounds = self.policy_round_names(start_year=start_year, end_year=end_year)
        observations: list[PointInTimeFeatureObservation] = []
        for round_name in rounds:
            for item in requested:
                payload = self.fetch_series_payload(item.series_id, policy_round=round_name)
                observations.extend(
                    self.observations_from_payload(
                        payload,
                        feature_name=item.feature_name,
                        series_id=item.series_id,
                        kind=item.kind,
                    )
                )
                if include_forecasts:
                    observations.extend(
                        self.forecast_observations_from_payload(
                            payload,
                            feature_name=item.feature_name,
                            series_id=item.series_id,
                            kind=item.kind,
                            horizons=forecast_horizons,
                        )
                    )
        return self._store(observations)


__all__ = [
    "BASE_URL",
    "DEFAULT_FORECAST_HORIZONS",
    "DEFAULT_SERIES",
    "RiksbankForecastHorizon",
    "RiksbankMonetaryPolicyProvider",
    "RiksbankSeries",
    "policy_round_names",
]
