from __future__ import annotations

from dataclasses import dataclass
import math
import re
from xml.etree import ElementTree as ET

import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.providers.http import ProviderError


SOURCE_NAME = "bolagsverket-ixbrl"
_XBRLI_NS = "http://www.xbrl.org/2003/instance"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"


@dataclass(frozen=True)
class BolagsverketDocument:
    document_id: str
    file_format: str
    report_period_end: pd.Timestamp
    registered_at: pd.Timestamp


@dataclass(frozen=True)
class BolagsverketFactSpec:
    feature_name: str
    concept_names: tuple[str, ...]
    required_currency: str | None = "SEK"
    kind: AuxiliaryFeatureKind = AuxiliaryFeatureKind.FUNDAMENTAL

    def __post_init__(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("Bolagsverket feature_name cannot be empty")
        if not self.concept_names or any(not value.strip() for value in self.concept_names):
            raise ValueError("Bolagsverket fact spec requires concept names")
        if self.required_currency is not None and not self.required_currency.strip():
            raise ValueError("required_currency cannot be blank")


DEFAULT_FACT_SPECS = (
    BolagsverketFactSpec("revenue", ("Nettoomsattning",)),
    BolagsverketFactSpec("operating_income", ("Rorelseresultat",)),
    BolagsverketFactSpec("net_income", ("AretsResultat",)),
    BolagsverketFactSpec("total_assets", ("Tillgangar",)),
    BolagsverketFactSpec("equity", ("EgetKapital",)),
    BolagsverketFactSpec("cash", ("KassaBank", "KassaBankExklRedovisningsmedel")),
    BolagsverketFactSpec("long_term_liabilities", ("LangfristigaSkulder",)),
    BolagsverketFactSpec("current_liabilities", ("KortfristigaSkulder",)),
)


def _utc(value, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ProviderError(f"Bolagsverket returned invalid {label}") from exc
    if pd.isna(timestamp):
        raise ProviderError(f"Bolagsverket returned invalid {label}")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _row_value(row: dict, *names: str):
    lowered = {str(key).casefold(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.casefold())
        if value not in (None, ""):
            return value
    return None


def _document_rows(payload) -> list[dict]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("dokument", "documents", "items", "results", "data"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            known = {str(key).casefold() for key in payload}
            if "dokumentid" in known or "documentid" in known:
                rows = [payload]
            else:
                raise ProviderError("Bolagsverket returned unsupported document-list payload shape")
    else:
        raise ProviderError("Bolagsverket returned unsupported document-list payload type")
    if not rows:
        raise ProviderError("Bolagsverket returned no documents")
    if not all(isinstance(row, dict) for row in rows):
        raise ProviderError("Bolagsverket returned malformed document metadata")
    return list(rows)


def parse_document_list(payload) -> tuple[BolagsverketDocument, ...]:
    """Parse Bolagsverket document metadata without inferring availability.

    `REGISTRERINGSTIDPUNKT` is the source timestamp at which the document became
    registered at Bolagsverket and is therefore used as the point-in-time availability
    boundary for all facts extracted from that filing.
    """

    documents: list[BolagsverketDocument] = []
    seen: set[str] = set()
    for row in _document_rows(payload):
        document_id = str(
            _row_value(row, "DOKUMENTID", "documentId", "document_id") or ""
        ).strip()
        file_format = str(
            _row_value(row, "FILFORMAT", "fileFormat", "file_format") or ""
        ).strip()
        raw_period = _row_value(
            row,
            "RAPPORTERINGSPERIODTOM",
            "reportingPeriodTo",
            "report_period_end",
        )
        raw_registered = _row_value(
            row,
            "REGISTRERINGSTIDPUNKT",
            "registrationTime",
            "registeredAt",
            "registered_at",
        )
        if not document_id:
            raise ProviderError("Bolagsverket document metadata lacks DOKUMENTID")
        if document_id in seen:
            raise ProviderError("Bolagsverket document list contains duplicate DOKUMENTID")
        if not file_format:
            raise ProviderError("Bolagsverket document metadata lacks FILFORMAT")
        if raw_period is None:
            raise ProviderError("Bolagsverket document metadata lacks RAPPORTERINGSPERIODTOM")
        if raw_registered is None:
            raise ProviderError("Bolagsverket document metadata lacks REGISTRERINGSTIDPUNKT")

        report_period_end = _utc(raw_period, label="reporting period end").normalize()
        registered_at = _utc(raw_registered, label="registration time")
        if registered_at < report_period_end:
            raise ProviderError("Bolagsverket registration time precedes reporting period end")
        seen.add(document_id)
        documents.append(
            BolagsverketDocument(
                document_id=document_id,
                file_format=file_format,
                report_period_end=report_period_end,
                registered_at=registered_at,
            )
        )

    return tuple(sorted(documents, key=lambda item: (item.registered_at, item.document_id)))


def _local_name(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag.split(":", 1)[-1]


def _concept_name(element: ET.Element) -> str:
    local = _local_name(element.tag)
    if local.casefold() == "nonfraction":
        raw = str(element.attrib.get("name", "")).strip()
        if not raw:
            raise ProviderError("Inline XBRL numeric fact lacks name")
        return raw.split(":", 1)[-1]
    return local


def _context_period_end(context: ET.Element) -> pd.Timestamp:
    instant = context.find(f".//{{{_XBRLI_NS}}}instant")
    if instant is not None and instant.text:
        return _utc(instant.text.strip(), label="XBRL context instant").normalize()
    end_date = context.find(f".//{{{_XBRLI_NS}}}endDate")
    if end_date is not None and end_date.text:
        return _utc(end_date.text.strip(), label="XBRL context end date").normalize()
    raise ProviderError("XBRL context lacks instant/endDate")


def _contexts(root: ET.Element) -> dict[str, pd.Timestamp]:
    output: dict[str, pd.Timestamp] = {}
    for element in root.iter():
        if _local_name(element.tag).casefold() != "context":
            continue
        context_id = str(element.attrib.get("id", "")).strip()
        if not context_id:
            raise ProviderError("XBRL context lacks id")
        if context_id in output:
            raise ProviderError("XBRL contains duplicate context id")
        output[context_id] = _context_period_end(element)
    if not output:
        raise ProviderError("XBRL document contains no contexts")
    return output


def _units(root: ET.Element) -> dict[str, str]:
    output: dict[str, str] = {}
    for element in root.iter():
        if _local_name(element.tag).casefold() != "unit":
            continue
        unit_id = str(element.attrib.get("id", "")).strip()
        if not unit_id:
            raise ProviderError("XBRL unit lacks id")
        measures = [
            str(child.text).strip()
            for child in element.iter()
            if _local_name(child.tag).casefold() == "measure" and child.text
        ]
        if len(measures) != 1:
            raise ProviderError("StockBot only supports simple single-measure XBRL units")
        if unit_id in output:
            raise ProviderError("XBRL contains duplicate unit id")
        output[unit_id] = measures[0]
    return output


def _normalize_numeric_text(text: str, format_name: str | None) -> str:
    raw = text.replace("\u00a0", " ").strip()
    if not raw:
        raise ProviderError("XBRL numeric fact is empty")
    format_local = None if not format_name else format_name.split(":", 1)[-1].casefold()
    if format_local in (None, "numdotdecimal"):
        normalized = raw.replace(" ", "").replace("'", "").replace(",", "")
    elif format_local == "numcommadecimal":
        normalized = raw.replace(" ", "").replace("'", "").replace(".", "").replace(",", ".")
    elif format_local in ("zerodash", "numdash") and raw in ("-", "–", "—"):
        return "0"
    else:
        raise ProviderError(f"unsupported Inline XBRL numeric format: {format_name}")
    return normalized


def _numeric_fact(element: ET.Element) -> float | None:
    nil = str(element.attrib.get(f"{{{_XSI_NS}}}nil", "false")).casefold()
    if nil in ("true", "1"):
        return None
    text = "".join(element.itertext())
    normalized = _normalize_numeric_text(text, element.attrib.get("format"))
    try:
        value = float(normalized)
    except ValueError as exc:
        raise ProviderError("XBRL numeric fact is not parseable") from exc
    scale_raw = element.attrib.get("scale")
    if scale_raw not in (None, ""):
        try:
            scale = int(scale_raw)
        except ValueError as exc:
            raise ProviderError("Inline XBRL scale is not an integer") from exc
        value *= 10.0 ** scale
    sign = str(element.attrib.get("sign", "")).strip()
    if sign:
        if sign != "-":
            raise ProviderError("unsupported Inline XBRL sign")
        value = -value
    if not math.isfinite(value):
        raise ProviderError("XBRL numeric fact is not finite")
    return float(value)


def _unit_matches(measure: str, currency: str | None) -> bool:
    if currency is None:
        return True
    value = str(measure).strip().upper()
    expected = currency.strip().upper()
    return value == expected or value.endswith(f":{expected}")


def build_feature_store_from_xbrl(
    content: bytes | str,
    *,
    document: BolagsverketDocument,
    symbol: str,
    fact_specs: tuple[BolagsverketFactSpec, ...] | list[BolagsverketFactSpec] = DEFAULT_FACT_SPECS,
) -> PointInTimeFeatureStore:
    """Extract configured point-in-time fundamentals from XBRL or Inline XBRL.

    Facts are revision-aware: comparative periods repeated in a newer annual report are
    emitted again with the newer `available_time`, allowing the feature store to expose
    old values before registration and revised values only after registration.
    """

    ticker = str(symbol or "").strip().upper()
    specs = tuple(fact_specs)
    if not ticker:
        raise ProviderError("Bolagsverket XBRL ingestion requires a StockBot symbol")
    if not specs:
        raise ProviderError("at least one Bolagsverket fact spec is required")
    feature_names = [item.feature_name.strip() for item in specs]
    if len(set(feature_names)) != len(feature_names):
        raise ProviderError("Bolagsverket fact feature names must be unique")

    raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    if not raw:
        raise ProviderError("Bolagsverket XBRL document is empty")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ProviderError("Bolagsverket document is not valid XML/XHTML") from exc

    contexts = _contexts(root)
    units = _units(root)
    spec_by_concept: dict[str, BolagsverketFactSpec] = {}
    for spec in specs:
        for concept in spec.concept_names:
            key = concept.strip().casefold()
            if key in spec_by_concept and spec_by_concept[key] != spec:
                raise ProviderError("Bolagsverket concept is mapped to multiple features")
            spec_by_concept[key] = spec

    selected: dict[tuple[str, pd.Timestamp], tuple[float, str, str]] = {}
    for element in root.iter():
        context_ref = str(element.attrib.get("contextRef", "")).strip()
        if not context_ref:
            continue
        concept = _concept_name(element)
        spec = spec_by_concept.get(concept.casefold())
        if spec is None:
            continue
        if context_ref not in contexts:
            raise ProviderError(f"XBRL fact references unknown context: {context_ref}")
        unit_ref = str(element.attrib.get("unitRef", "")).strip()
        if spec.required_currency is not None:
            if not unit_ref or unit_ref not in units:
                raise ProviderError(f"XBRL monetary fact {concept} lacks a known unit")
            if not _unit_matches(units[unit_ref], spec.required_currency):
                continue
        value = _numeric_fact(element)
        if value is None:
            continue
        observation_time = contexts[context_ref]
        key = (spec.feature_name, observation_time)
        candidate = (value, concept, context_ref)
        existing = selected.get(key)
        if existing is not None and existing[0] != value:
            raise ProviderError(
                f"ambiguous Bolagsverket fact for {spec.feature_name} at {observation_time.date()}"
            )
        if existing is None:
            selected[key] = candidate

    if not selected:
        raise ProviderError("Bolagsverket document contained no configured usable facts")

    observations: list[PointInTimeFeatureObservation] = []
    for (feature_name, observation_time), (value, concept, context_ref) in sorted(
        selected.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        if observation_time > document.report_period_end:
            raise ProviderError("XBRL fact context ends after document reporting period")
        observations.append(
            PointInTimeFeatureObservation(
                feature_name=feature_name,
                value=value,
                observation_time=observation_time.isoformat(),
                available_time=document.registered_at.isoformat(),
                source=SOURCE_NAME,
                kind=next(spec.kind for spec in specs if spec.feature_name == feature_name),
                symbol=ticker,
                revision_id=(
                    f"document={document.document_id}|concept={concept}|context={context_ref}"
                ),
            )
        )

    return PointInTimeFeatureStore(
        observations,
        PointInTimeFeatureManifest(
            source=SOURCE_NAME,
            point_in_time=True,
            revision_aware=True,
            available_time_semantics="bolagsverket_registration_time",
        ),
    )


def xbrl_content_from_document_zip(
    content: bytes,
    *,
    max_members: int = 64,
    max_uncompressed_bytes: int = 50_000_000,
    max_document_bytes: int = 20_000_000,
) -> bytes:
    """Extract one XBRL/iXBRL candidate from a bounded in-memory ZIP archive."""

    import io
    import zipfile

    if max_members <= 0 or max_uncompressed_bytes <= 0 or max_document_bytes <= 0:
        raise ValueError("Bolagsverket ZIP safety limits must be positive")
    raw = bytes(content)
    if not raw:
        raise ProviderError("Bolagsverket document ZIP is empty")

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw), mode="r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ProviderError("Bolagsverket document is not a valid zip archive") from exc

    with archive:
        members = archive.infolist()
        if len(members) > max_members:
            raise ProviderError("Bolagsverket document ZIP contains too many members")
        if any(info.flag_bits & 0x1 for info in members):
            raise ProviderError("encrypted Bolagsverket ZIP members are not supported")

        total_size = sum(int(info.file_size) for info in members if not info.is_dir())
        if total_size > max_uncompressed_bytes:
            raise ProviderError("Bolagsverket document ZIP exceeds uncompressed size limit")

        allowed_suffixes = (".xhtml", ".html", ".xbrl", ".xml")
        candidates = [
            info
            for info in members
            if not info.is_dir() and info.filename.casefold().endswith(allowed_suffixes)
        ]
        if not candidates:
            raise ProviderError("Bolagsverket document ZIP contains no XBRL candidate")
        if len(candidates) > 1:
            raise ProviderError("Bolagsverket document ZIP contains multiple XBRL candidates")

        candidate = candidates[0]
        if int(candidate.file_size) > max_document_bytes:
            raise ProviderError("Bolagsverket XBRL document exceeds size limit")
        try:
            extracted = archive.read(candidate)
        except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
            raise ProviderError("Bolagsverket XBRL ZIP member could not be read") from exc
        if len(extracted) > max_document_bytes:
            raise ProviderError("Bolagsverket XBRL document exceeds size limit")
        if not extracted:
            raise ProviderError("Bolagsverket XBRL ZIP member is empty")
        return extracted


def build_feature_store_from_document_zip(
    content: bytes,
    *,
    document: BolagsverketDocument,
    symbol: str,
    fact_specs: tuple[BolagsverketFactSpec, ...] | list[BolagsverketFactSpec] = DEFAULT_FACT_SPECS,
    max_members: int = 64,
    max_uncompressed_bytes: int = 50_000_000,
    max_document_bytes: int = 20_000_000,
) -> PointInTimeFeatureStore:
    """Build a point-in-time fundamental store directly from Bolagsverket ZIP bytes."""

    if "zip" not in document.file_format.casefold():
        raise ProviderError("Bolagsverket document metadata does not describe a ZIP file")
    xbrl = xbrl_content_from_document_zip(
        content,
        max_members=max_members,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_document_bytes=max_document_bytes,
    )
    return build_feature_store_from_xbrl(
        xbrl,
        document=document,
        symbol=symbol,
        fact_specs=fact_specs,
    )


def build_company_feature_store_from_document_zips(
    filings: tuple[tuple[BolagsverketDocument, bytes], ...]
    | list[tuple[BolagsverketDocument, bytes]],
    *,
    symbol: str,
    fact_specs: tuple[BolagsverketFactSpec, ...] | list[BolagsverketFactSpec] = DEFAULT_FACT_SPECS,
    max_members: int = 64,
    max_uncompressed_bytes: int = 50_000_000,
    max_document_bytes: int = 20_000_000,
) -> PointInTimeFeatureStore:
    """Combine multiple registered annual-report vintages without rewriting history."""

    entries = tuple(filings)
    if not entries:
        raise ProviderError("at least one Bolagsverket filing is required")
    document_ids = [document.document_id for document, _ in entries]
    if any(not str(document_id).strip() for document_id in document_ids):
        raise ProviderError("Bolagsverket filing document IDs cannot be blank")
    if len(set(document_ids)) != len(document_ids):
        raise ProviderError("Bolagsverket filing document IDs must be unique")

    observations: list[PointInTimeFeatureObservation] = []
    for document, archive in sorted(
        entries,
        key=lambda item: (item[0].registered_at, item[0].document_id),
    ):
        store = build_feature_store_from_document_zip(
            archive,
            document=document,
            symbol=symbol,
            fact_specs=fact_specs,
            max_members=max_members,
            max_uncompressed_bytes=max_uncompressed_bytes,
            max_document_bytes=max_document_bytes,
        )
        observations.extend(store.observations)

    return PointInTimeFeatureStore(
        observations,
        PointInTimeFeatureManifest(
            source=SOURCE_NAME,
            point_in_time=True,
            revision_aware=True,
            available_time_semantics="bolagsverket_registration_time",
        ),
    )
