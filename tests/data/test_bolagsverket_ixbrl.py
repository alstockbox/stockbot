import io
import zipfile

import pandas as pd
import pytest

from stockbot.data.providers.bolagsverket_ixbrl import (
    DEFAULT_FACT_SPECS,
    build_feature_store_from_document_zip,
    build_feature_store_from_xbrl,
    parse_document_list,
)
from stockbot.data.providers.http import ProviderError


def _document_payload():
    return {
        "dokument": [
            {
                "DOKUMENTID": "doc-2025-1",
                "FILFORMAT": "application/zip",
                "RAPPORTERINGSPERIODTOM": "2025-12-31",
                "REGISTRERINGSTIDPUNKT": "2026-06-30T10:15:00+02:00",
            }
        ]
    }


def _xbrl_bytes():
    return b'''<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:iso4217="http://www.xbrl.org/2003/iso4217"
            xmlns:se="http://www.taxonomier.se/se/fr/gen-base/2021-10-31">
  <xbrli:context id="period0">
    <xbrli:entity><xbrli:identifier scheme="http://www.bolagsverket.se">556999-9999</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:context id="balans0">
    <xbrli:entity><xbrli:identifier scheme="http://www.bolagsverket.se">556999-9999</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period>
  </xbrli:context>
  <xbrli:unit id="SEK"><xbrli:measure>iso4217:SEK</xbrli:measure></xbrli:unit>
  <se:Nettoomsattning contextRef="period0" unitRef="SEK" decimals="INF">12500000</se:Nettoomsattning>
  <se:Rorelseresultat contextRef="period0" unitRef="SEK" decimals="INF">2100000</se:Rorelseresultat>
  <se:AretsResultat contextRef="period0" unitRef="SEK" decimals="INF">1550000</se:AretsResultat>
  <se:Tillgangar contextRef="balans0" unitRef="SEK" decimals="INF">18400000</se:Tillgangar>
  <se:EgetKapital contextRef="balans0" unitRef="SEK" decimals="INF">9200000</se:EgetKapital>
  <se:KassaBank contextRef="balans0" unitRef="SEK" decimals="INF">3100000</se:KassaBank>
  <se:LangfristigaSkulder contextRef="balans0" unitRef="SEK" decimals="INF">3600000</se:LangfristigaSkulder>
  <se:KortfristigaSkulder contextRef="balans0" unitRef="SEK" decimals="INF">5600000</se:KortfristigaSkulder>
</xbrli:xbrl>'''


def _document_zip(*, duplicate_candidate: bool = False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", "Bolagsverket fixture")
        archive.writestr("annual-report.xhtml", _xbrl_bytes())
        if duplicate_candidate:
            archive.writestr("second-report.xbrl", _xbrl_bytes())
    return buffer.getvalue()


def test_document_list_preserves_registration_time_as_point_in_time_availability():
    documents = parse_document_list(_document_payload())

    assert len(documents) == 1
    document = documents[0]
    assert document.document_id == "doc-2025-1"
    assert document.file_format == "application/zip"
    assert document.report_period_end == pd.Timestamp("2025-12-31", tz="UTC")
    assert document.registered_at == pd.Timestamp("2026-06-30T08:15:00Z")


def test_xbrl_fundamentals_materialize_with_report_period_and_registration_time():
    document = parse_document_list(_document_payload())[0]
    store = build_feature_store_from_xbrl(
        _xbrl_bytes(),
        document=document,
        symbol="AAA",
        fact_specs=DEFAULT_FACT_SPECS,
    )

    expected = {
        "revenue": 12_500_000.0,
        "operating_income": 2_100_000.0,
        "net_income": 1_550_000.0,
        "total_assets": 18_400_000.0,
        "equity": 9_200_000.0,
        "cash": 3_100_000.0,
        "long_term_liabilities": 3_600_000.0,
        "current_liabilities": 5_600_000.0,
    }
    assert set(store.feature_names) == set(expected)
    assert store.manifest.source == "bolagsverket-ixbrl"
    assert store.manifest.point_in_time is True
    assert store.manifest.revision_aware is True
    assert store.manifest.available_time_semantics == "bolagsverket_registration_time"

    asof = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-06-30T08:14:59Z"), "AAA"),
            (pd.Timestamp("2026-06-30T08:15:00Z"), "AAA"),
        ],
        names=["timestamp", "symbol"],
    )
    frame = store.materialize(asof, feature_names=tuple(expected))
    assert frame.iloc[0].isna().all()
    for feature_name, value in expected.items():
        assert frame.iloc[1][feature_name] == value

    for observation in store.observations:
        assert observation.observation_time == "2025-12-31T00:00:00+00:00"
        assert observation.available_time == "2026-06-30T08:15:00+00:00"
        assert observation.symbol == "AAA"
        assert "document=doc-2025-1" in str(observation.revision_id)


def test_document_zip_builds_same_point_in_time_store_without_extracting_to_disk():
    document = parse_document_list(_document_payload())[0]
    direct = build_feature_store_from_xbrl(
        _xbrl_bytes(), document=document, symbol="AAA", fact_specs=DEFAULT_FACT_SPECS
    )
    zipped = build_feature_store_from_document_zip(
        _document_zip(), document=document, symbol="AAA", fact_specs=DEFAULT_FACT_SPECS
    )

    assert zipped.fingerprint == direct.fingerprint


def test_document_zip_rejects_ambiguous_multiple_xbrl_candidates():
    document = parse_document_list(_document_payload())[0]
    with pytest.raises(ProviderError, match="multiple XBRL"):
        build_feature_store_from_document_zip(
            _document_zip(duplicate_candidate=True),
            document=document,
            symbol="AAA",
            fact_specs=DEFAULT_FACT_SPECS,
        )


def test_registration_time_before_reporting_period_is_rejected():
    payload = _document_payload()
    payload["dokument"][0]["REGISTRERINGSTIDPUNKT"] = "2025-12-30T12:00:00Z"
    with pytest.raises(ProviderError, match="precedes reporting period"):
        parse_document_list(payload)


def test_conflicting_same_period_fact_is_rejected_as_ambiguous():
    document = parse_document_list(_document_payload())[0]
    conflicting = _xbrl_bytes().replace(
        b"</xbrli:xbrl>",
        b'<se:Nettoomsattning contextRef="period0" unitRef="SEK" decimals="INF">999</se:Nettoomsattning></xbrli:xbrl>',
    )
    with pytest.raises(ProviderError, match="ambiguous Bolagsverket fact"):
        build_feature_store_from_xbrl(
            conflicting,
            document=document,
            symbol="AAA",
            fact_specs=DEFAULT_FACT_SPECS,
        )
