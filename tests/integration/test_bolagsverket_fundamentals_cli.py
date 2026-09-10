import io
import json
import zipfile

import pytest

from stockbot.cli.bolagsverket_fundamentals import build_parser, run_from_args
from stockbot.data.bolagsverket_research_input import build_store_from_manifest
from stockbot.data.providers.http import ProviderError
from stockbot.data.research_inputs import load_point_in_time_feature_store


def _xbrl(organization_number="556999-9999"):
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:iso4217="http://www.xbrl.org/2003/iso4217"
            xmlns:se="http://www.taxonomier.se/se/fr/gen-base/2021-10-31">
  <xbrli:context id="p2024"><xbrli:entity><xbrli:identifier scheme="http://www.bolagsverket.se">{organization_number}</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="p2025"><xbrli:entity><xbrli:identifier scheme="http://www.bolagsverket.se">{organization_number}</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:unit id="SEK"><xbrli:measure>iso4217:SEK</xbrli:measure></xbrli:unit>
  <se:Nettoomsattning contextRef="p2024" unitRef="SEK">100</se:Nettoomsattning>
  <se:Nettoomsattning contextRef="p2025" unitRef="SEK">125</se:Nettoomsattning>
</xbrli:xbrl>'''.encode("utf-8")


def _write_zip(path, content):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annual-report.xhtml", content)


def _write_manifest(tmp_path, *, organization_number="556999-9999"):
    zip_path = tmp_path / "aaa-2025.zip"
    _write_zip(zip_path, _xbrl())
    manifest = {
        "schema_version": 1,
        "companies": [
            {
                "symbol": "AAA",
                "organization_number": organization_number,
                "filings": [
                    {
                        "document_id": "doc-2025",
                        "file_format": "application/zip",
                        "report_period_end": "2025-12-31",
                        "registered_at": "2026-06-30T08:15:00Z",
                        "zip_path": zip_path.name,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_bolagsverket_cli_builds_fingerprinted_factory_input_with_derived_features(tmp_path):
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "bolagsverket-features.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--include-derived",
        ]
    )

    assert run_from_args(args) == 0
    loaded = load_point_in_time_feature_store(output)
    assert "revenue" in loaded.feature_names
    assert "revenue_yoy" in loaded.feature_names
    assert loaded.manifest.point_in_time is True
    assert loaded.manifest.revision_aware is True
    assert loaded.fingerprint


def test_manifest_rejects_symbol_to_wrong_organization_number_mapping(tmp_path):
    manifest = _write_manifest(tmp_path, organization_number="556111-1111")

    with pytest.raises(ProviderError, match="organization number"):
        build_store_from_manifest(manifest, include_derived=False)
