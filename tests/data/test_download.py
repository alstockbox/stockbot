import pandas as pd
import pytest

from stockbot.data.download import DownloadError, download_market_snapshot
from stockbot.data.providers.http import ProviderError
from stockbot.data.schemas import DataGrade
from stockbot.data.snapshots import SnapshotStore


class FakeProvider:
    name = "fixture"
    grade = DataGrade.BOOTSTRAP
    def __init__(self, fail=None): self.fail = fail
    def fetch_bars(self, symbol, start, end):
        if symbol == self.fail:
            raise ProviderError("boom")
        dt = pd.Timestamp("2026-01-02", tz="UTC")
        px = 10.0 if symbol == "AAA" else 20.0
        return pd.DataFrame([{
            "timestamp": dt, "symbol": symbol,
            "open": px, "high": px+1, "low": px-1, "close": px+0.5, "volume": 1000,
            "adj_open": px, "adj_high": px+1, "adj_low": px-1, "adj_close": px+0.5, "adj_volume": 1000,
            "div_cash": 0.0, "split_factor": 1.0, "provider": self.name,
            "retrieved_at": pd.Timestamp("2026-01-03", tz="UTC"),
        }])


def test_download_sorts_deduplicates_universe_and_persists_complete_snapshot(tmp_path):
    snapshot = download_market_snapshot(
        FakeProvider(), ["BBB", "aaa", "AAA"], "2026-01-01", "2026-01-31", SnapshotStore(tmp_path)
    )
    assert snapshot.manifest.symbols == ("AAA", "BBB")
    assert set(snapshot.bars["symbol"]) == {"AAA", "BBB"}
    assert snapshot.manifest.grade is DataGrade.BOOTSTRAP


def test_download_fails_closed_when_any_requested_symbol_fails(tmp_path):
    store = SnapshotStore(tmp_path)
    with pytest.raises(DownloadError):
        download_market_snapshot(FakeProvider(fail="BBB"), ["AAA", "BBB"], "2026-01-01", "2026-01-31", store)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("symbols,start,end", [([], "2026-01-01", "2026-01-31"), (["AAA"], "2026-02-01", "2026-01-31")])
def test_download_rejects_invalid_requests(tmp_path, symbols, start, end):
    with pytest.raises(ValueError):
        download_market_snapshot(FakeProvider(), symbols, start, end, SnapshotStore(tmp_path))


def test_bootstrap_provider_cannot_be_upgraded_to_research_grade(tmp_path):
    with pytest.raises(DownloadError):
        download_market_snapshot(
            FakeProvider(), ["AAA"], "2026-01-01", "2026-01-31", SnapshotStore(tmp_path),
            grade=DataGrade.RESEARCH_GRADE,
            provenance={"point_in_time_universe": True},
        )
