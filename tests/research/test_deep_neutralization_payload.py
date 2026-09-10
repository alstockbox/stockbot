from types import SimpleNamespace

from stockbot.research.deep_diagnostics import _compact_neutralization


def _exposure(max_weight, worst, hhi, active, unclassified, coverage=1.0):
    return SimpleNamespace(
        sector_coverage=coverage,
        average_max_sector_weight=max_weight,
        worst_max_sector_weight=worst,
        average_sector_hhi=hhi,
        average_active_sectors=active,
        average_unclassified_weight=unclassified,
    )


def test_compact_neutralization_includes_executed_sector_concentration():
    report = SimpleNamespace(
        baseline_score=1.2,
        neutralized_score=1.4,
        score_delta=0.2,
        sector_coverage=0.99,
        prediction_coverage=0.85,
        average_abs_sector_mean_before=0.03,
        average_abs_sector_mean_after=0.01,
        factor_correlations_before={"momentum_20": 0.3},
        factor_correlations_after={"momentum_20": 0.05},
        neutralized_metrics={"sharpe": 1.1, "cagr": 0.12},
        neutralized_stress=SimpleNamespace(score=0.8),
        baseline_sector_exposure=_exposure(0.72, 0.90, 0.58, 2.0, 0.01, 0.99),
        neutralized_sector_exposure=_exposure(0.44, 0.60, 0.36, 3.0, 0.00, 1.0),
    )

    payload = _compact_neutralization(report)

    assert payload["executed_sector_coverage_before"] == 0.99
    assert payload["executed_sector_coverage_after"] == 1.0
    assert payload["average_max_sector_weight_before"] == 0.72
    assert payload["average_max_sector_weight_after"] == 0.44
    assert payload["worst_max_sector_weight_before"] == 0.90
    assert payload["worst_max_sector_weight_after"] == 0.60
    assert payload["sector_hhi_before"] == 0.58
    assert payload["sector_hhi_after"] == 0.36
    assert payload["active_sectors_before"] == 2.0
    assert payload["active_sectors_after"] == 3.0
    assert payload["unclassified_weight_before"] == 0.01
    assert payload["unclassified_weight_after"] == 0.00
