from __future__ import annotations

from collections import defaultdict
import math

import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)


DERIVED_FUNDAMENTAL_FEATURES = (
    "revenue_yoy",
    "operating_margin",
    "net_margin",
    "equity_ratio",
    "cash_to_assets",
    "liabilities_to_assets",
    "return_on_assets",
    "return_on_equity",
    "asset_growth_yoy",
)

_MIN_ANNUAL_GAP_DAYS = 300
_MAX_ANNUAL_GAP_DAYS = 430


def _document_id(revision_id: str | None) -> str | None:
    if revision_id is None:
        return None
    for token in str(revision_id).split("|"):
        key, separator, value = token.partition("=")
        if separator and key.strip().casefold() == "document" and value.strip():
            return value.strip()
    return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def _fact_map(
    observations: list[PointInTimeFeatureObservation],
) -> dict[tuple[str, pd.Timestamp], PointInTimeFeatureObservation]:
    facts: dict[tuple[str, pd.Timestamp], PointInTimeFeatureObservation] = {}
    for observation in observations:
        timestamp = pd.Timestamp(observation.observation_time)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        key = (observation.feature_name, timestamp)
        existing = facts.get(key)
        if existing is not None and float(existing.value) != float(observation.value):
            raise ValueError(
                "ambiguous raw fundamental fact inside one filing vintage: "
                f"{observation.feature_name} at {timestamp.isoformat()}"
            )
        if existing is None:
            facts[key] = observation
    return facts


def _append_ratio(
    output: list[PointInTimeFeatureObservation],
    *,
    facts: dict[tuple[str, pd.Timestamp], PointInTimeFeatureObservation],
    document_id: str,
    feature_name: str,
    observation_time: pd.Timestamp,
    numerator_name: str,
    denominator_name: str,
    source: str,
    symbol: str,
    available_time: str,
) -> None:
    numerator = facts.get((numerator_name, observation_time))
    denominator = facts.get((denominator_name, observation_time))
    if numerator is None or denominator is None:
        return
    value = _safe_ratio(float(numerator.value), float(denominator.value))
    if value is None:
        return
    output.append(
        PointInTimeFeatureObservation(
            feature_name=feature_name,
            value=value,
            observation_time=observation_time.isoformat(),
            available_time=available_time,
            source=source,
            kind=AuxiliaryFeatureKind.FUNDAMENTAL,
            symbol=symbol,
            revision_id=(
                f"document={document_id}|derived={feature_name}|"
                f"inputs={numerator_name},{denominator_name}"
            ),
        )
    )


def _append_liability_ratio(
    output: list[PointInTimeFeatureObservation],
    *,
    facts: dict[tuple[str, pd.Timestamp], PointInTimeFeatureObservation],
    document_id: str,
    observation_time: pd.Timestamp,
    source: str,
    symbol: str,
    available_time: str,
) -> None:
    long_term = facts.get(("long_term_liabilities", observation_time))
    current = facts.get(("current_liabilities", observation_time))
    assets = facts.get(("total_assets", observation_time))
    if long_term is None or current is None or assets is None:
        return
    value = _safe_ratio(
        float(long_term.value) + float(current.value),
        float(assets.value),
    )
    if value is None:
        return
    output.append(
        PointInTimeFeatureObservation(
            feature_name="liabilities_to_assets",
            value=value,
            observation_time=observation_time.isoformat(),
            available_time=available_time,
            source=source,
            kind=AuxiliaryFeatureKind.FUNDAMENTAL,
            symbol=symbol,
            revision_id=(
                f"document={document_id}|derived=liabilities_to_assets|"
                "inputs=long_term_liabilities,current_liabilities,total_assets"
            ),
        )
    )


def _append_annual_growth(
    output: list[PointInTimeFeatureObservation],
    *,
    facts: dict[tuple[str, pd.Timestamp], PointInTimeFeatureObservation],
    document_id: str,
    raw_feature: str,
    derived_feature: str,
    source: str,
    symbol: str,
    available_time: str,
) -> None:
    periods = sorted(
        timestamp
        for feature_name, timestamp in facts
        if feature_name == raw_feature
    )
    for previous_time, current_time in zip(periods, periods[1:]):
        gap_days = (current_time - previous_time).total_seconds() / 86400.0
        if not _MIN_ANNUAL_GAP_DAYS <= gap_days <= _MAX_ANNUAL_GAP_DAYS:
            continue
        previous = facts[(raw_feature, previous_time)]
        current = facts[(raw_feature, current_time)]
        value = _safe_ratio(float(current.value), float(previous.value))
        if value is None:
            continue
        growth = value - 1.0
        if not math.isfinite(growth):
            continue
        output.append(
            PointInTimeFeatureObservation(
                feature_name=derived_feature,
                value=growth,
                observation_time=current_time.isoformat(),
                available_time=available_time,
                source=source,
                kind=AuxiliaryFeatureKind.FUNDAMENTAL,
                symbol=symbol,
                revision_id=(
                    f"document={document_id}|derived={derived_feature}|"
                    f"input={raw_feature}|prior={previous_time.isoformat()}"
                ),
            )
        )


def add_derived_fundamentals(store: PointInTimeFeatureStore) -> PointInTimeFeatureStore:
    """Return a store augmented with causal, filing-vintage derived fundamentals.

    Derivations are performed independently inside each source filing. This prevents a
    ratio from mixing facts belonging to different registration vintages and ensures a
    later restatement can only alter derived features after the later filing's own
    `available_time`.
    """

    if not store.manifest.point_in_time or not store.manifest.revision_aware:
        raise ValueError("derived fundamentals require point-in-time revision-aware data")
    existing_derived = set(store.feature_names).intersection(DERIVED_FUNDAMENTAL_FEATURES)
    if existing_derived:
        raise ValueError(
            "fundamental store already contains derived feature names: "
            + ",".join(sorted(existing_derived))
        )

    groups: dict[
        tuple[str, str, str, str],
        list[PointInTimeFeatureObservation],
    ] = defaultdict(list)
    for observation in store.observations:
        if observation.kind is not AuxiliaryFeatureKind.FUNDAMENTAL:
            continue
        document_id = _document_id(observation.revision_id)
        if document_id is None or observation.symbol is None:
            continue
        groups[
            (
                observation.symbol,
                observation.available_time,
                observation.source,
                document_id,
            )
        ].append(observation)

    if not groups:
        raise ValueError("no filing-identified fundamental observations available for derivation")

    derived: list[PointInTimeFeatureObservation] = []
    for (symbol, available_time, source, document_id), observations in sorted(groups.items()):
        facts = _fact_map(observations)
        periods = sorted({timestamp for _, timestamp in facts})
        for observation_time in periods:
            _append_ratio(
                derived,
                facts=facts,
                document_id=document_id,
                feature_name="operating_margin",
                observation_time=observation_time,
                numerator_name="operating_income",
                denominator_name="revenue",
                source=source,
                symbol=symbol,
                available_time=available_time,
            )
            _append_ratio(
                derived,
                facts=facts,
                document_id=document_id,
                feature_name="net_margin",
                observation_time=observation_time,
                numerator_name="net_income",
                denominator_name="revenue",
                source=source,
                symbol=symbol,
                available_time=available_time,
            )
            _append_ratio(
                derived,
                facts=facts,
                document_id=document_id,
                feature_name="equity_ratio",
                observation_time=observation_time,
                numerator_name="equity",
                denominator_name="total_assets",
                source=source,
                symbol=symbol,
                available_time=available_time,
            )
            _append_ratio(
                derived,
                facts=facts,
                document_id=document_id,
                feature_name="cash_to_assets",
                observation_time=observation_time,
                numerator_name="cash",
                denominator_name="total_assets",
                source=source,
                symbol=symbol,
                available_time=available_time,
            )
            _append_liability_ratio(
                derived,
                facts=facts,
                document_id=document_id,
                observation_time=observation_time,
                source=source,
                symbol=symbol,
                available_time=available_time,
            )
            _append_ratio(
                derived,
                facts=facts,
                document_id=document_id,
                feature_name="return_on_assets",
                observation_time=observation_time,
                numerator_name="net_income",
                denominator_name="total_assets",
                source=source,
                symbol=symbol,
                available_time=available_time,
            )
            _append_ratio(
                derived,
                facts=facts,
                document_id=document_id,
                feature_name="return_on_equity",
                observation_time=observation_time,
                numerator_name="net_income",
                denominator_name="equity",
                source=source,
                symbol=symbol,
                available_time=available_time,
            )

        _append_annual_growth(
            derived,
            facts=facts,
            document_id=document_id,
            raw_feature="revenue",
            derived_feature="revenue_yoy",
            source=source,
            symbol=symbol,
            available_time=available_time,
        )
        _append_annual_growth(
            derived,
            facts=facts,
            document_id=document_id,
            raw_feature="total_assets",
            derived_feature="asset_growth_yoy",
            source=source,
            symbol=symbol,
            available_time=available_time,
        )

    if not derived:
        raise ValueError("fundamental filing data produced no valid derived features")

    return PointInTimeFeatureStore(
        tuple(store.observations) + tuple(derived),
        store.manifest,
    )
