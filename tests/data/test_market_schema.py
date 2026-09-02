import pandas as pd
import pytest

from stockbot.data.market_schema import CANONICAL_BAR_COLUMNS, validate_canonical_bars
from stockbot.data.schemas import DataGrade


def _bars():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-02"], utc=True),
        "symbol": ["AAA"],
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "volume": [1000],
        "adj_open": [5.0], "adj_high": [5.5], "adj_low": [4.5], "adj_close": [5.25], "adj_volume": [2000],
        "div_cash": [0.25], "split_factor": [2.0],
        "provider": ["fixture"],
        "retrieved_at": pd.to_datetime(["2026-01-03"], utc=True),
    })


def test_data_grade_includes_bootstrap_without_breaking_research_grade():
    assert DataGrade.BOOTSTRAP.value == "bootstrap"
    assert DataGrade.RESEARCH_GRADE.value == "research_grade"


def test_canonical_schema_preserves_raw_and_adjusted_values():
    frame = _bars()
    validate_canonical_bars(frame)
    assert set(CANONICAL_BAR_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "close"] == 10.5
    assert frame.loc[0, "adj_close"] == 5.25


@pytest.mark.parametrize("mutator", [
    lambda f: f.assign(high=[8.0]),
    lambda f: f.assign(low=[12.0]),
    lambda f: f.assign(volume=[-1]),
    lambda f: pd.concat([f, f], ignore_index=True),
])
def test_canonical_schema_rejects_invalid_market_rows(mutator):
    with pytest.raises(ValueError):
        validate_canonical_bars(mutator(_bars()))
