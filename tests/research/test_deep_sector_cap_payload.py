from types import SimpleNamespace

from stockbot.research.deep_diagnostics import _compact_sector_cap


def _exposure(*, coverage, average_max, worst_max, hhi, active, unclassified):
    return SimpleNamespace(
        sector_coverage=coverage,
        average_max_sector_weight=average_max,
        worst_max_sector_weight=worst_max,
        average_sector_hhi=hhi,
        average_active_sectors=active,
        average_unclassified_weight=unclassified,
    )


def test_compact_sector_cap_reports_performance_and_executed_concentration_delta():
    report = SimpleNamespace(
        max_sector_weight=0.35,
        baseline_score=1.2,
        constrained_score=1.4,
        score_delta=0.2,
        baseline_metrics={"sharpe": 1.1, "cagr": 0.12},
        constrained_metrics={"sharpe": 1.3, "cagr": 0.14},
        baseline_stress=SimpleNamespace(score=0.60),
        constrained_stress=SimpleNamespace(score=0.72),
        baseline_sector_exposure=_exposure(
            coverage=0.99,
            average_max=0.58,
            worst_max=0.82,
            hhi=0.44,
            active=2.1,
            unclassified=0.01,
        ),
        constrained_sector_exposure=_exposure(
            coverage=0.995,
            average_max=0.34,
            worst_max=0.39,
            hhi=0.29,
            active=3.2,
            unclassified=0.005,
        ),
        average_signal_max_sector_weight_before=0.57,
        average_signal_max_sector_weight_after=0.33,
        average_cash_weight=0.04,
    )

    payload = _compact_sector_cap(report)

    assert payload["max_sector_weight"] == 0.35
    assert payload["baseline_score"] == 1.2
    assert payload["constrained_score"] == 1.4
    assert payload["score_delta"] == 0.2
    assert payload["baseline_sharpe"] == 1.1
    assert payload["constrained_sharpe"] == 1.3
    assert payload["baseline_cagr"] == 0.12
    assert payload["constrained_cagr"] == 0.14
    assert payload["baseline_stress"] == 0.60
    assert payload["constrained_stress"] == 0.72
    assert payload["average_signal_max_sector_weight_before"] == 0.57
    assert payload["average_signal_max_sector_weight_after"] == 0.33
    assert payload["average_cash_weight"] == 0.04
    assert payload["executed_sector_coverage_before"] == 0.99
    assert payload["executed_sector_coverage_after"] == 0.995
    assert payload["average_max_sector_weight_before"] == 0.58
    assert payload["average_max_sector_weight_after"] == 0.34
    assert payload["worst_max_sector_weight_before"] == 0.82
    assert payload["worst_max_sector_weight_after"] == 0.39
    assert payload["sector_hhi_before"] == 0.44
    assert payload["sector_hhi_after"] == 0.29
    assert payload["active_sectors_before"] == 2.1
    assert payload["active_sectors_after"] == 3.2
    assert payload["unclassified_weight_before"] == 0.01
    assert payload["unclassified_weight_after"] == 0.005

    assert "signal_diagnostics" not in payload
    assert "constrained_signal_weights" not in payload
    assert "constrained_executed_weights" not in payload
    assert "constrained_net_returns" not in payload
    assert "constrained_turnover" not in payload
