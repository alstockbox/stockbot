import io
import zipfile

import pandas as pd

from stockbot.data.providers.bolagsverket_ixbrl import (
    BolagsverketDocument,
    build_company_feature_store_from_document_zips,
    build_feature_store_from_document_zip,
)


def _xbrl(periods):
    context_rows = []
    fact_rows = []
    for suffix, period_end, revenue in periods:
        context_rows.append(
            f'''<xbrli:context id="p{suffix}">
  <xbrli:entity><xbrli:identifier scheme="http://www.bolagsverket.se">556999-9999</xbrli:identifier></xbrli:entity>
  <xbrli:period><xbrli:startDate>{period_end[:4]}-01-01</xbrli:startDate><xbrli:endDate>{period_end}</xbrli:endDate></xbrli:period>
</xbrli:context>'''
        )
        fact_rows.append(
            f'<se:Nettoomsattning contextRef="p{suffix}" unitRef="SEK" decimals="INF">{revenue}</se:Nettoomsattning>'
        )
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:iso4217="http://www.xbrl.org/2003/iso4217"
            xmlns:se="http://www.taxonomier.se/se/fr/gen-base/2021-10-31">
{''.join(context_rows)}
<xbrli:unit id="SEK"><xbrli:measure>iso4217:SEK</xbrli:measure></xbrli:unit>
{''.join(fact_rows)}
</xbrli:xbrl>'''
    return xml.encode("utf-8")


def _zip(xbrl):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annual-report.xhtml", xbrl)
    return buffer.getvalue()


def _document(document_id, period_end, registered_at):
    return BolagsverketDocument(
        document_id=document_id,
        file_format="application/zip",
        report_period_end=pd.Timestamp(period_end, tz="UTC"),
        registered_at=pd.Timestamp(registered_at),
    )


def test_future_filing_cannot_rewrite_earlier_materialized_fundamental_history():
    filing_2024 = _document("doc-2024", "2024-12-31", "2025-06-01T08:00:00Z")
    filing_2025 = _document("doc-2025", "2025-12-31", "2026-06-01T08:00:00Z")
    zip_2024 = _zip(_xbrl((("2024", "2024-12-31", 100.0),)))
    zip_2025 = _zip(
        _xbrl(
            (
                ("2024", "2024-12-31", 110.0),
                ("2025", "2025-12-31", 150.0),
            )
        )
    )

    old_store = build_feature_store_from_document_zip(
        zip_2024, document=filing_2024, symbol="AAA"
    )
    combined = build_company_feature_store_from_document_zips(
        ((filing_2024, zip_2024), (filing_2025, zip_2025)),
        symbol="AAA",
    )

    historical_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-12-01T00:00:00Z"), "AAA")],
        names=["timestamp", "symbol"],
    )
    pd.testing.assert_frame_equal(
        old_store.materialize(historical_index, feature_names=("revenue",)),
        combined.materialize(historical_index, feature_names=("revenue",)),
    )
    assert combined.materialize(
        historical_index, feature_names=("revenue",)
    ).iloc[0, 0] == 100.0

    revenue_2024_vintages = sorted(
        (
            observation
            for observation in combined.observations
            if observation.feature_name == "revenue"
            and observation.observation_time.startswith("2024-12-31")
        ),
        key=lambda observation: observation.available_time,
    )
    assert [row.value for row in revenue_2024_vintages] == [100.0, 110.0]
    assert [row.available_time for row in revenue_2024_vintages] == [
        "2025-06-01T08:00:00+00:00",
        "2026-06-01T08:00:00+00:00",
    ]

    after_new_filing = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-06-02T00:00:00Z"), "AAA")],
        names=["timestamp", "symbol"],
    )
    assert combined.materialize(
        after_new_filing, feature_names=("revenue",)
    ).iloc[0, 0] == 150.0
