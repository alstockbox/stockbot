import pandas as pd

from stockbot.data.providers.bolagsverket_ixbrl import (
    BolagsverketDocument,
    build_feature_store_from_xbrl,
)


def _document():
    return BolagsverketDocument(
        document_id="doc-2025",
        file_format="application/zip",
        report_period_end=pd.Timestamp("2025-12-31", tz="UTC"),
        registered_at=pd.Timestamp("2026-06-30T08:15:00Z"),
    )


def _xbrl():
    return b'''<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:iso4217="http://www.xbrl.org/2003/iso4217"
            xmlns:se="http://www.taxonomier.se/se/fr/gen-base/2021-10-31">
  <xbrli:context id="p2024"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="p2025"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="b2024"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period></xbrli:context>
  <xbrli:context id="b2025"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period></xbrli:context>
  <xbrli:unit id="SEK"><xbrli:measure>iso4217:SEK</xbrli:measure></xbrli:unit>

  <se:Nettoomsattning contextRef="p2024" unitRef="SEK">100</se:Nettoomsattning>
  <se:Tillgangar contextRef="b2024" unitRef="SEK">400</se:Tillgangar>

  <se:Nettoomsattning contextRef="p2025" unitRef="SEK">125</se:Nettoomsattning>
  <se:Rorelseresultat contextRef="p2025" unitRef="SEK">25</se:Rorelseresultat>
  <se:AretsResultat contextRef="p2025" unitRef="SEK">15</se:AretsResultat>
  <se:Tillgangar contextRef="b2025" unitRef="SEK">500</se:Tillgangar>
  <se:EgetKapital contextRef="b2025" unitRef="SEK">200</se:EgetKapital>
  <se:KassaBank contextRef="b2025" unitRef="SEK">50</se:KassaBank>
  <se:LangfristigaSkulder contextRef="b2025" unitRef="SEK">120</se:LangfristigaSkulder>
  <se:KortfristigaSkulder contextRef="b2025" unitRef="SEK">180</se:KortfristigaSkulder>
</xbrli:xbrl>'''


def test_derived_fundamentals_are_computed_inside_same_filing_vintage():
    store = build_feature_store_from_xbrl(
        _xbrl(),
        document=_document(),
        symbol="AAA",
        include_derived_fundamentals=True,
    )

    expected = {
        "revenue_yoy": 0.25,
        "operating_margin": 0.20,
        "net_margin": 0.12,
        "equity_ratio": 0.40,
        "cash_to_assets": 0.10,
        "liabilities_to_assets": 0.60,
        "return_on_assets": 0.03,
        "return_on_equity": 0.075,
        "asset_growth_yoy": 0.25,
    }
    derived = {
        observation.feature_name: observation
        for observation in store.observations
        if observation.feature_name in expected
    }
    assert set(derived) == set(expected)
    for name, value in expected.items():
        assert derived[name].value == value
        assert derived[name].observation_time == "2025-12-31T00:00:00+00:00"
        assert derived[name].available_time == "2026-06-30T08:15:00+00:00"
        assert derived[name].symbol == "AAA"
        assert "document=doc-2025" in str(derived[name].revision_id)
        assert "derived=" in str(derived[name].revision_id)

    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-06-30T08:14:59Z"), "AAA"),
            (pd.Timestamp("2026-06-30T08:15:00Z"), "AAA"),
        ],
        names=["timestamp", "symbol"],
    )
    frame = store.materialize(index, feature_names=tuple(expected))
    assert frame.iloc[0].isna().all()
    for name, value in expected.items():
        assert frame.iloc[1][name] == value


def test_zero_denominator_does_not_emit_infinite_derived_features():
    zero_revenue = _xbrl().replace(
        b'<se:Nettoomsattning contextRef="p2025" unitRef="SEK">125</se:Nettoomsattning>',
        b'<se:Nettoomsattning contextRef="p2025" unitRef="SEK">0</se:Nettoomsattning>',
    )
    store = build_feature_store_from_xbrl(
        zero_revenue,
        document=_document(),
        symbol="AAA",
        include_derived_fundamentals=True,
    )

    names = set(store.feature_names)
    assert "operating_margin" not in names
    assert "net_margin" not in names
    assert "revenue_yoy" not in names
