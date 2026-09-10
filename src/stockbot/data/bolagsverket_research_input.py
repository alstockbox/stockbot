from __future__ import annotations

import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from stockbot.data.point_in_time_features import (
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.providers.bolagsverket_ixbrl import (
    SOURCE_NAME,
    BolagsverketDocument,
    build_company_feature_store_from_document_zips,
    parse_document_list,
    xbrl_content_from_document_zip,
)
from stockbot.data.providers.http import ProviderError
from stockbot.features.fundamentals import add_derived_fundamentals


BOLAGSVERKET_MANIFEST_SCHEMA_VERSION = 1
_XBRLI_NS = "http://www.xbrl.org/2003/instance"


def _normalize_organization_number(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) == 12 and digits.startswith("16"):
        digits = digits[2:]
    if len(digits) != 10:
        raise ProviderError("Bolagsverket organization number must contain 10 digits")
    return digits


def _xbrl_organization_number(content: bytes) -> str:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ProviderError("Bolagsverket document is not valid XML/XHTML") from exc

    identifiers: set[str] = set()
    for context in root.iter(f"{{{_XBRLI_NS}}}context"):
        identifier = context.find(f".//{{{_XBRLI_NS}}}identifier")
        if identifier is None or not identifier.text or not identifier.text.strip():
            raise ProviderError("Bolagsverket XBRL context lacks organization number")
        identifiers.add(_normalize_organization_number(identifier.text))

    if not identifiers:
        raise ProviderError("Bolagsverket XBRL document contains no organization number")
    if len(identifiers) != 1:
        raise ProviderError("Bolagsverket XBRL contexts contain multiple organization numbers")
    return next(iter(identifiers))


def _read_manifest(path: str | Path) -> tuple[Path, dict[str, object]]:
    target = Path(path).expanduser().resolve()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderError("Bolagsverket manifest could not be read") from exc
    if not isinstance(raw, dict):
        raise ProviderError("Bolagsverket manifest must be a JSON object")
    try:
        schema_version = int(raw.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise ProviderError("Bolagsverket manifest has invalid schema version") from exc
    if schema_version != BOLAGSVERKET_MANIFEST_SCHEMA_VERSION:
        raise ProviderError("unsupported Bolagsverket manifest schema version")
    return target, raw


def _resolve_zip_path(manifest_path: Path, raw_path: object) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ProviderError("Bolagsverket filing manifest lacks zip_path")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ProviderError(f"Bolagsverket filing ZIP does not exist: {candidate}")
    return candidate


def _documents_and_archives(
    manifest_path: Path,
    company: dict[str, object],
) -> tuple[tuple[tuple[BolagsverketDocument, bytes], ...], str, str]:
    symbol = str(company.get("symbol") or "").strip().upper()
    if not symbol:
        raise ProviderError("Bolagsverket company manifest lacks symbol")
    expected_org = _normalize_organization_number(company.get("organization_number"))

    raw_filings = company.get("filings")
    if not isinstance(raw_filings, list) or not raw_filings:
        raise ProviderError("Bolagsverket company manifest requires filings")

    metadata_rows: list[dict[str, object]] = []
    zip_paths: dict[str, Path] = {}
    for raw in raw_filings:
        if not isinstance(raw, dict):
            raise ProviderError("Bolagsverket filing manifest entry must be an object")
        document_id = str(raw.get("document_id") or "").strip()
        if not document_id:
            raise ProviderError("Bolagsverket filing manifest lacks document_id")
        if document_id in zip_paths:
            raise ProviderError("Bolagsverket filing manifest contains duplicate document_id")
        metadata_rows.append(
            {
                "DOKUMENTID": document_id,
                "FILFORMAT": raw.get("file_format"),
                "RAPPORTERINGSPERIODTOM": raw.get("report_period_end"),
                "REGISTRERINGSTIDPUNKT": raw.get("registered_at"),
            }
        )
        zip_paths[document_id] = _resolve_zip_path(manifest_path, raw.get("zip_path"))

    documents = parse_document_list(metadata_rows)
    filings: list[tuple[BolagsverketDocument, bytes]] = []
    for document in documents:
        try:
            archive = zip_paths[document.document_id].read_bytes()
        except OSError as exc:
            raise ProviderError("Bolagsverket filing ZIP could not be read") from exc
        xbrl = xbrl_content_from_document_zip(archive)
        actual_org = _xbrl_organization_number(xbrl)
        if actual_org != expected_org:
            raise ProviderError(
                "Bolagsverket organization number mismatch between manifest and XBRL: "
                f"symbol={symbol} expected={expected_org} actual={actual_org}"
            )
        filings.append((document, archive))

    return tuple(filings), symbol, expected_org


def build_store_from_manifest(
    path: str | Path,
    *,
    include_derived: bool = False,
) -> PointInTimeFeatureStore:
    """Build one fingerprintable point-in-time fundamental store from local filings.

    Every manifest symbol is bound to the organization number present in each filing's
    XBRL contexts before any fact is admitted. Registration timestamps remain the
    availability boundary used by the underlying Bolagsverket provider.
    """

    manifest_path, manifest = _read_manifest(path)
    companies = manifest.get("companies")
    if not isinstance(companies, list) or not companies:
        raise ProviderError("Bolagsverket manifest requires companies")

    observations: list[PointInTimeFeatureObservation] = []
    seen_symbols: set[str] = set()
    seen_document_ids: set[str] = set()

    for raw_company in companies:
        if not isinstance(raw_company, dict):
            raise ProviderError("Bolagsverket company manifest entry must be an object")
        filings, symbol, _ = _documents_and_archives(manifest_path, raw_company)
        if symbol in seen_symbols:
            raise ProviderError("Bolagsverket manifest contains duplicate symbol")
        seen_symbols.add(symbol)

        for document, _ in filings:
            if document.document_id in seen_document_ids:
                raise ProviderError("Bolagsverket manifest document_id must be globally unique")
            seen_document_ids.add(document.document_id)

        company_store = build_company_feature_store_from_document_zips(
            filings,
            symbol=symbol,
        )
        if include_derived:
            company_store = add_derived_fundamentals(company_store)
        observations.extend(company_store.observations)

    return PointInTimeFeatureStore(
        tuple(observations),
        PointInTimeFeatureManifest(
            source=SOURCE_NAME,
            point_in_time=True,
            revision_aware=True,
            available_time_semantics="bolagsverket_registration_time",
        ),
    )
