import io
import zipfile

import pandas as pd

from stockbot.data.providers.bolagsverket_ixbrl import (
    BolagsverketDocument,
    build_company_feature_store_from_document_zips,
)
from stockbot.features.fundamentals import add_derived_fundamentals


def _document(document_id, period_end, registered_at):
    return BolagsverketDocument(
        document_id=document_id,
        file_format="application/zip",
        report_period_end=pd.Timestamp(period_end, tz="UTC"),
        registered_at=pd.Timestamp(registered_at),
    )


def _xbrl(periods):
    contexts = []
    facts = []
    for suffix, end_date, revenue in periods:
        year = end_date[:4]
        contexts.append(
            f'''<xbrli:context id="p{suffix}"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>{year}-01-01</xbrli:startDate><xbrli:endDate>{end_date}</xbrli:endDate></xbrli:period></xbrli:context>'''
        )
        facts.append(
            f'<se:Nettoomsattning contextRef="p{suffix}" unitRef="SEK">{revenue}</se:Nettoomsattning>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:iso4217="http://www.xbrl.org/2003/iso4217"
            xmlns:se="http://www.taxonomier.se/se/fr/gen-base/2021-10-31">
{''.join(contexts)}
<xbrli:unit id="SEK"><xbrli:measure>iso4217:SEK</xbrli:measure></xbrli:unit>
{''.join(facts)}
</xbrli:xbrl>'''.encode("utf-8")


def _zip(content):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annual-report.xhtml", content)
    return buffer.getvalue()


def test_future_filing_can_revise_derived_growth_only_after_its_registration_time():
    filing_2024 = _document("doc-2024", "2024-12-31", "2025-06-01T08:00:00Z")
    filing_2025 = _document("doc-2025", "2025-12-31", "2026-06-01T08:00:00Z")

    raw = build_company_feature_store_from_document_zips(
        (
            (
                filing_2024,
                _zip(
                    _xbrl(
                        (
                            ("2023", "2023-12-31", 80.0),
                            ("2024", "2024-12-31", 100.0),
                        )
                    )
                ),
            ),
            (
                filing_2025,
                _zip(
                    _xbrl(
                        (
                            ("2024", "2024-12-31", 110.0),
                            ("2025", "2025-12-31", 150.0),
                        )
                    )
                ),
            ),
        ),
        symbol="AAA",
    )
    store = add_derived_fundamentals(raw)

    growth_observations = sorted(
        (
            observation
            for observation in store.observations
            if observation.feature_name == "revenue_yoy"
        ),
        key=lambda observation: observation.available_time,
    )
    assert len(growth_observations) == 2
    assert growth_observations[0].value == 0.25
    assert growth_observations[0].observation_time == "2024-12-31T00:00:00+00:00"
    assert growth_observations[0].available_time == "2025-06-01T08:00:00+00:00"
    assert abs(growth_observations[1].value - (150.0 / 110.0 - 1.0)) < 1e-12
    assert growth_observations[1].observation_time == "2025-12-31T00:00:00+00:00"
    assert growth_observations[1].available_time == "2026-06-01T08:00:00+00:00"

    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-12-01T00:00:00Z"), "AAA"),
            (pd.Timestamp("2026-06-02T00:00:00Z"), "AAA"),
        ],
        names=["timestamp", "symbol"],
    )
    frame = store.materialize(index, feature_names=("revenue_yoy",))
    assert frame.iloc[0, 0] == 0.25
    assert abs(frame.iloc[1, 0] - (150.0 / 110.0 - 1.0)) < 1e-12
