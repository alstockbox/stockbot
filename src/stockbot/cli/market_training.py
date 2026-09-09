from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from stockbot.data.download import DownloadError, download_market_snapshot
from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.tiingo import TiingoProvider
from stockbot.data.providers.yahoo_bootstrap import YahooBootstrapProvider
from stockbot.data.research_inputs import (
    load_point_in_time_feature_store,
    load_point_in_time_universe,
)
from stockbot.data.research_quality import (
    ResearchDataAttestation,
    evaluate_research_data_quality,
    research_data_quality_fingerprint,
    verified_data_grade,
)
from stockbot.data.snapshots import SnapshotStore
from stockbot.research.deep_diagnostics import compact_deep_diagnostics
from stockbot.research.factory import ResearchFactoryConfig
from stockbot.research.jobs import make_job_manifest, make_run_summary, write_json_payload, write_json_record
from stockbot.research.liquidity_execution import LiquidityExecutionConfig
from stockbot.research.market_training import (
    run_snapshot_deep_diagnostics,
    run_snapshot_factory,
    run_snapshot_regime_specialists,
    train_snapshot,
)
from stockbot.research.population import ModelPopulationConfig
from stockbot.research.quarantine import QuarantineConfig, load_quarantine_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download real EOD market data into an immutable StockBot snapshot and "
            "optionally run the V1 Alpha Arena or the V2 multi-horizon research factory."
        )
    )
    parser.add_argument("--provider", required=True, choices=("tiingo", "yahoo-bootstrap"))
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT,SPY")
    parser.add_argument("--start", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--snapshot-root", default="snapshots", help="Directory for immutable snapshots")
    parser.add_argument("--train", action="store_true", help="Run the existing V1 Alpha Arena after download")
    parser.add_argument("--horizon", type=int, default=5, help="Forward-return horizon used by V1 training")
    parser.add_argument("--factory", action="store_true", help="Run the V2 autonomous research factory")
    parser.add_argument("--factory-horizons", default="1,5,20", help="Comma-separated forward-return horizons for V2")
    parser.add_argument("--factory-candidates", type=int, default=160, help="Maximum ML challenger configurations explored per horizon")
    parser.add_argument("--factory-workers", type=int, default=4, help="Parallel challenger workers used by V2")
    parser.add_argument("--factory-memory", default="research_memory/experiments.jsonl", help="Append-only research-memory JSONL path")
    parser.add_argument("--factory-run-dir", default=None, help="Optional directory for scheduler-friendly job manifest and research artifacts")
    parser.add_argument(
        "--factory-auxiliary-input",
        default=None,
        help="Fingerprint-verified StockBot point-in-time auxiliary feature artifact",
    )
    parser.add_argument(
        "--factory-auxiliary-features",
        default=None,
        help="Optional comma-separated subset of auxiliary feature names",
    )
    parser.add_argument(
        "--factory-auxiliary-max-age-days",
        type=int,
        default=None,
        help="Optional maximum age in days for point-in-time auxiliary observations",
    )
    parser.add_argument(
        "--factory-auxiliary-min-coverage",
        type=float,
        default=0.80,
        help="Minimum materialized auxiliary feature-cell coverage required for research",
    )
    parser.add_argument(
        "--factory-universe-input",
        default=None,
        help="Fingerprint-verified StockBot point-in-time universe artifact used by neutralization/data controls",
    )
    parser.add_argument(
        "--factory-attest-adjusted-prices-verified",
        action="store_true",
        help="Explicitly attest that adjusted historical prices were verified by the ingestion pipeline",
    )
    parser.add_argument(
        "--factory-attest-corporate-actions-complete",
        action="store_true",
        help="Explicitly attest that split/dividend corporate actions are complete for the research dataset",
    )
    parser.add_argument(
        "--factory-attest-corporate-actions-point-in-time",
        action="store_true",
        help="Explicitly attest that corporate-action history is point-in-time correct",
    )
    parser.add_argument(
        "--factory-quarantine-start",
        default=None,
        help="Explicit sealed audit boundary; rows at/after this date are excluded from all routine research",
    )
    parser.add_argument(
        "--factory-quarantine-min-development-periods",
        type=int,
        default=252,
        help="Minimum development periods required before the sealed quarantine boundary",
    )
    parser.add_argument(
        "--factory-quarantine-min-periods",
        type=int,
        default=42,
        help="Minimum periods required inside the sealed quarantine set",
    )
    parser.add_argument("--factory-regime-specialists", action="store_true", help="Train diagnostic bull/bear/chop specialists on the same development data")
    parser.add_argument("--factory-specialists-top-k", type=int, default=2, help="Generalist candidates per horizon admitted to regime-specialist research")
    parser.add_argument("--factory-deep-diagnostics", action="store_true", help="Run expensive pre-holdout policy/window/feature/liquidity diagnostics")
    parser.add_argument("--factory-deep-top-k", type=int, default=1, help="Generalist candidates per horizon admitted to deep diagnostics")
    parser.add_argument("--factory-train-windows", default="126,252,504", help="Comma-separated walk-forward training windows for deep diagnostics")
    parser.add_argument("--factory-diagnostic-test-periods", type=int, default=21, help="OOS test-window length used by deep diagnostics")
    parser.add_argument("--factory-diagnostic-capital", type=float, default=100_000.0, help="Reference portfolio capital used for the single liquidity stress")
    parser.add_argument(
        "--factory-diagnostic-capital-grid",
        default="10000,25000,50000,100000,250000,500000,1000000,2500000,5000000",
        help="Comma-separated capital levels used to estimate the strategy capacity curve",
    )
    parser.add_argument("--factory-diagnostic-max-participation", type=float, default=0.05, help="Maximum fraction of lagged ADV that may be traded per symbol/day")
    parser.add_argument("--factory-diagnostic-spread-bps", type=float, default=4.0, help="Full execution spread assumption in basis points")
    parser.add_argument("--factory-diagnostic-impact-bps", type=float, default=10.0, help="Square-root market impact in bps at 1 percent ADV participation")
    parser.add_argument("--factory-diagnostic-commission-bps", type=float, default=1.0, help="Commission assumption in basis points")
    parser.add_argument("--factory-diagnostic-adv-window", type=int, default=20, help="Lagged ADV rolling window used by liquidity simulation")
    return parser


def resolve_provider(name: str):
    if name == "yahoo-bootstrap":
        return YahooBootstrapProvider()
    if name == "tiingo":
        token = os.environ.get("TIINGO_API_TOKEN", "").strip()
        if not token:
            raise ProviderError("TIINGO_API_TOKEN is required for --provider tiingo")
        return TiingoProvider(token=token)
    raise ValueError(f"unsupported provider: {name}")


def _parse_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_names(value: str | None, label: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError(f"{label} cannot be empty")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must be unique")
    return parsed


def _parse_positive_ints(value: str, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{label} must contain positive integers")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must be unique")
    return parsed


def _parse_positive_floats(value: str, label: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated numbers") from exc
    if not parsed or any(item <= 0.0 for item in parsed):
        raise ValueError(f"{label} must contain positive values")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must be unique")
    return parsed


def _parse_horizons(value: str) -> tuple[int, ...]:
    return _parse_positive_ints(value, "factory horizons")


def _evaluate_factory_data_quality(snapshot, point_in_time_universe, args):
    attestation = ResearchDataAttestation(
        adjusted_prices_verified=bool(args.factory_attest_adjusted_prices_verified),
        corporate_actions_complete=bool(args.factory_attest_corporate_actions_complete),
        corporate_actions_point_in_time=bool(args.factory_attest_corporate_actions_point_in_time),
    )
    report = evaluate_research_data_quality(
        snapshot.bars,
        universe=point_in_time_universe,
        attestation=attestation,
    )
    fingerprint = research_data_quality_fingerprint(report)
    effective_grade = verified_data_grade(snapshot.manifest.grade, report)
    return report, fingerprint, effective_grade


def run_from_args(args: argparse.Namespace) -> int:
    provider = resolve_provider(args.provider)
    symbols = _parse_symbols(args.symbols)
    store = SnapshotStore(Path(args.snapshot_root))
    snapshot = download_market_snapshot(provider, symbols, args.start, args.end, store)
    manifest = snapshot.manifest
    print(f"snapshot_id={snapshot.snapshot_id}")
    print(f"provider={manifest.provider}")
    print(f"grade={manifest.grade.value}")
    print(f"symbols={','.join(manifest.symbols)}")
    print(f"rows={manifest.row_count}")
    print(f"fingerprint={manifest.dataset_fingerprint}")
    print(f"path={snapshot.path}")

    if args.train:
        run = train_snapshot(snapshot, horizon=args.horizon)
        print("alpha_arena:")
        for index, result in enumerate(run.leaderboard, start=1):
            print(
                f"  {index}. {result.name} score={result.score:.6f} "
                f"oos={result.oos_coverage:.3f} drawdown={result.metrics.get('max_drawdown', float('nan')):.4f}"
            )
        champion = run.champion_candidate.name if run.champion_candidate is not None else "none"
        print(f"champion_candidate={champion}")

    if args.factory:
        horizons = _parse_horizons(args.factory_horizons)
        train_windows = _parse_positive_ints(args.factory_train_windows, "factory train windows")
        capital_levels = _parse_positive_floats(args.factory_diagnostic_capital_grid, "factory diagnostic capital grid")
        auxiliary_feature_names = _parse_names(args.factory_auxiliary_features, "factory auxiliary features")
        if args.factory_specialists_top_k <= 0:
            raise ValueError("factory specialists top-k must be positive")
        if args.factory_deep_top_k <= 0:
            raise ValueError("factory deep top-k must be positive")
        if args.factory_diagnostic_test_periods <= 0:
            raise ValueError("factory diagnostic test periods must be positive")
        if args.factory_diagnostic_capital <= 0:
            raise ValueError("factory diagnostic capital must be positive")
        if not 0.0 < args.factory_diagnostic_max_participation <= 1.0:
            raise ValueError("factory diagnostic max participation must be in (0,1]")
        if min(
            args.factory_diagnostic_spread_bps,
            args.factory_diagnostic_impact_bps,
            args.factory_diagnostic_commission_bps,
        ) < 0:
            raise ValueError("factory diagnostic execution costs cannot be negative")
        if args.factory_diagnostic_adv_window <= 0:
            raise ValueError("factory diagnostic ADV window must be positive")
        if args.factory_quarantine_min_development_periods <= 0 or args.factory_quarantine_min_periods <= 0:
            raise ValueError("factory quarantine period requirements must be positive")
        if args.factory_auxiliary_max_age_days is not None and args.factory_auxiliary_max_age_days < 0:
            raise ValueError("factory auxiliary max age days cannot be negative")
        if not 0.0 < args.factory_auxiliary_min_coverage <= 1.0:
            raise ValueError("factory auxiliary minimum coverage must be in (0,1]")
        if auxiliary_feature_names is not None and not args.factory_auxiliary_input:
            raise ValueError("factory auxiliary features require --factory-auxiliary-input")

        auxiliary_store = (
            None
            if not args.factory_auxiliary_input
            else load_point_in_time_feature_store(args.factory_auxiliary_input)
        )
        point_in_time_universe = (
            None
            if not args.factory_universe_input
            else load_point_in_time_universe(args.factory_universe_input)
        )
        if auxiliary_store is not None and auxiliary_feature_names is not None:
            missing_auxiliary = sorted(set(auxiliary_feature_names).difference(auxiliary_store.feature_names))
            if missing_auxiliary:
                raise ValueError(f"unknown requested auxiliary features: {missing_auxiliary}")

        quarantine_config = None
        quarantine_manifest_path = None
        if args.factory_quarantine_start:
            quarantine_config = QuarantineConfig(
                start=args.factory_quarantine_start,
                min_development_periods=args.factory_quarantine_min_development_periods,
                min_quarantine_periods=args.factory_quarantine_min_periods,
            )
            quarantine_manifest_path = Path(args.factory_memory).with_suffix(".quarantine.json")

        selected_auxiliary_names = (
            ()
            if auxiliary_store is None
            else (
                auxiliary_store.feature_names
                if auxiliary_feature_names is None
                else auxiliary_feature_names
            )
        )
        config = ResearchFactoryConfig(
            horizons=horizons,
            population=ModelPopulationConfig(max_candidates=args.factory_candidates),
            max_workers=args.factory_workers,
        )
        job_manifest = make_job_manifest(
            snapshot_id=snapshot.snapshot_id,
            dataset_fingerprint=manifest.dataset_fingerprint,
            horizons=horizons,
            max_candidates=args.factory_candidates,
            max_workers=args.factory_workers,
            memory_path=args.factory_memory,
            quarantine_start=(None if quarantine_config is None else quarantine_config.start),
            auxiliary_fingerprint=(None if auxiliary_store is None else auxiliary_store.fingerprint),
            auxiliary_features=selected_auxiliary_names,
            universe_fingerprint=(
                None if point_in_time_universe is None else point_in_time_universe.fingerprint
            ),
        )
        run_dir = None
        if args.factory_run_dir:
            run_dir = Path(args.factory_run_dir) / job_manifest.job_id
            write_json_record(run_dir / "job.json", job_manifest)

        report = run_snapshot_factory(
            snapshot,
            config=config,
            memory_path=args.factory_memory,
            quarantine_config=quarantine_config,
            quarantine_manifest_path=quarantine_manifest_path,
            auxiliary_store=auxiliary_store,
            auxiliary_feature_names=auxiliary_feature_names,
            auxiliary_max_age_days=args.factory_auxiliary_max_age_days,
            auxiliary_min_coverage=args.factory_auxiliary_min_coverage,
        )

        specialist_diagnostics = None
        if args.factory_regime_specialists:
            specialist_diagnostics = run_snapshot_regime_specialists(
                snapshot,
                report,
                top_k_per_horizon=args.factory_specialists_top_k,
                quarantine_config=quarantine_config,
                quarantine_manifest_path=quarantine_manifest_path,
                auxiliary_store=auxiliary_store,
                auxiliary_feature_names=auxiliary_feature_names,
                auxiliary_max_age_days=args.factory_auxiliary_max_age_days,
                auxiliary_min_coverage=args.factory_auxiliary_min_coverage,
            )

        deep_diagnostics = None
        if args.factory_deep_diagnostics:
            liquidity_config = LiquidityExecutionConfig(
                capital=args.factory_diagnostic_capital,
                commission_bps=args.factory_diagnostic_commission_bps,
                spread_bps=args.factory_diagnostic_spread_bps,
                impact_bps_at_one_pct_adv=args.factory_diagnostic_impact_bps,
                max_participation=args.factory_diagnostic_max_participation,
                adv_window=args.factory_diagnostic_adv_window,
            )
            deep_diagnostics = run_snapshot_deep_diagnostics(
                snapshot,
                report,
                top_k_per_horizon=args.factory_deep_top_k,
                train_windows=train_windows,
                test_periods=args.factory_diagnostic_test_periods,
                liquidity_config=liquidity_config,
                capacity_levels=capital_levels,
                quarantine_config=quarantine_config,
                quarantine_manifest_path=quarantine_manifest_path,
                point_in_time_universe=point_in_time_universe,
                auxiliary_store=auxiliary_store,
                auxiliary_feature_names=auxiliary_feature_names,
                auxiliary_max_age_days=args.factory_auxiliary_max_age_days,
                auxiliary_min_coverage=args.factory_auxiliary_min_coverage,
            )

        quarantine_manifest = None
        if quarantine_manifest_path is not None:
            quarantine_manifest = load_quarantine_manifest(quarantine_manifest_path)

        if run_dir is not None:
            write_json_record(run_dir / "summary.json", make_run_summary(job_manifest.job_id, report, specialist_diagnostics))
            if deep_diagnostics is not None:
                write_json_payload(run_dir / "deep_diagnostics.json", compact_deep_diagnostics(deep_diagnostics))
            if quarantine_manifest is not None:
                write_json_payload(run_dir / "quarantine.json", asdict(quarantine_manifest))

        print("research_factory:")
        print(f"  job_id={job_manifest.job_id}")
        print(f"  experiments_run={report.experiments_run}")
        print(f"  promotion_candidates={report.candidates_passed}")
        print(f"  holdout_evaluated={report.holdout_evaluated}")
        print(f"  holdout_start={report.holdout_start}")
        if auxiliary_store is not None:
            print(f"  auxiliary_fingerprint={auxiliary_store.fingerprint}")
            print("  auxiliary_features=" + ",".join(selected_auxiliary_names))
        if point_in_time_universe is not None:
            print(f"  universe_fingerprint={point_in_time_universe.fingerprint}")
        if quarantine_manifest is not None:
            print(f"  quarantine_id={quarantine_manifest.quarantine_id}")
            print(f"  quarantine_start={quarantine_manifest.start}")
            print(f"  quarantine_periods={quarantine_manifest.quarantine_periods}")
            print(f"  quarantine_rows={quarantine_manifest.quarantine_rows}")
            print(f"  quarantine_manifest={quarantine_manifest_path}")

        for index, candidate in enumerate(report.candidates[:20], start=1):
            gate = "pass" if candidate.gate.passed else "reject"
            holdout = "not_tested" if candidate.holdout_report is None else ("pass" if candidate.holdout_report.passed else "reject")
            q_value = float("nan") if candidate.discovery is None else candidate.discovery.q_value
            regime_score = float("nan") if candidate.regime_report is None else candidate.regime_report.score
            drift = "unknown" if candidate.drift_report is None else ("degraded" if candidate.drift_report.degraded else "stable")
            readiness = "unknown" if candidate.paper_readiness is None else ("ready" if candidate.paper_readiness.ready else "not_ready")
            print(
                f"  {index}. h={candidate.horizon} {candidate.model_name} "
                f"research={candidate.factory_score:.6f} selection={candidate.selection_score:.6f} "
                f"promotion={candidate.promotion_score:.6f} oos={candidate.oos_coverage:.3f} "
                f"stress={candidate.stress_score:.3f} regime={regime_score:.3f} "
                f"q={q_value:.4f} drift={drift} readiness={readiness} gate={gate} holdout={holdout}"
            )

        if report.ensemble_report is not None:
            ensemble = report.ensemble_report
            print(
                f"horizon_ensemble_score={ensemble.score:.6f} "
                f"sharpe={ensemble.metrics.get('sharpe', float('nan')):.3f} stress={ensemble.stress_report.score:.3f}"
            )
            print("horizon_ensemble_weights=" + ",".join(f"{key}:{weight:.4f}" for key, weight in ensemble.member_weights.items()))

        if specialist_diagnostics is not None:
            print(f"regime_specialist_candidate_pairs={specialist_diagnostics.candidate_pairs_tested}")
            for regime_name, specialist in specialist_diagnostics.specialists.items():
                print(
                    f"regime_specialist={regime_name} h={specialist.horizon} model={specialist.model_config.name} "
                    f"score={specialist.score:.6f} oos={specialist.oos_coverage:.3f} stress={specialist.stress_report.score:.3f}"
                )
            if specialist_diagnostics.router_report is not None:
                router = specialist_diagnostics.router_report
                print(
                    f"regime_router_score={router.score:.6f} sharpe={router.metrics.get('sharpe', float('nan')):.3f} "
                    f"stress={router.stress_report.score:.3f}"
                )

        if deep_diagnostics is not None:
            print(f"deep_diagnostic_candidates={deep_diagnostics.candidate_count}")
            print(
                f"deep_liquidity_assumptions=capital:{args.factory_diagnostic_capital:.2f},"
                f"max_participation:{args.factory_diagnostic_max_participation:.4f},"
                f"spread_bps:{args.factory_diagnostic_spread_bps:.2f},"
                f"impact_bps:{args.factory_diagnostic_impact_bps:.2f}"
            )
            print("deep_capacity_grid=" + ",".join(f"{capital:.0f}" for capital in capital_levels))
            for experiment_id, diagnostic in deep_diagnostics.candidates.items():
                best_policy = diagnostic.policy_arena.best
                bootstrap = diagnostic.bootstrap_uncertainty
                factor = diagnostic.factor_exposure
                liquidity = diagnostic.liquidity_execution
                capacity = diagnostic.capacity_curve
                neutral = diagnostic.neutralization
                max_capacity = "none" if capacity.max_effective_capital is None else f"{capacity.max_effective_capital:.0f}"
                first_break = "none" if capacity.first_break_capital is None else f"{capacity.first_break_capital:.0f}"
                neutral_text = (
                    "neutralization=none"
                    if neutral is None
                    else (
                        f"neutral_score={neutral.neutralized_score:.6f} "
                        f"neutral_delta={neutral.score_delta:.6f} "
                        f"sector_bias={neutral.average_abs_sector_mean_before:.6f}->"
                        f"{neutral.average_abs_sector_mean_after:.6f}"
                    )
                )
                print(
                    f"deep={experiment_id} h={diagnostic.horizon} model={diagnostic.model_name} "
                    f"policy={best_policy.policy.top_fraction:.2f}/{best_policy.policy.weighting} "
                    f"policy_score={best_policy.score:.6f} confidence={bootstrap.confidence_score:.3f} "
                    f"p_cagr_pos={bootstrap.probability_positive_cagr:.3f} p_sharpe_pos={bootstrap.probability_positive_sharpe:.3f} "
                    f"residual_sharpe={factor.residual_metrics.get('sharpe', float('nan')):.3f} "
                    f"factor_r2={factor.r_squared:.3f} idio={factor.idiosyncratic_score:.3f} "
                    f"liquidity_score={liquidity.score:.6f} partial_fill={liquidity.partial_fill_fraction:.3f} "
                    f"tracking={liquidity.average_tracking_error:.3f} capacity_score={capacity.capacity_score:.3f} "
                    f"max_capacity={max_capacity} first_break={first_break} {neutral_text} "
                    f"window_score={diagnostic.window_robustness.score:.3f} "
                    f"drop_groups={','.join(diagnostic.feature_ablation.recommended_drop_groups) or 'none'}"
                )
            for horizon, stacking in sorted(deep_diagnostics.stacking_reports.items()):
                print(
                    f"stacking=h{horizon} members={len(stacking.member_ids)} model={stacking.meta_model.name} "
                    f"score={stacking.score:.6f} oos={stacking.oos_coverage:.3f} "
                    f"sharpe={stacking.metrics.get('sharpe', float('nan')):.3f} stress={stacking.stress_report.score:.3f}"
                )

        champion = report.champion_candidate
        if champion is None:
            print("factory_champion_candidate=none")
        else:
            print(f"factory_champion_candidate=h{champion.horizon}:{champion.model_name}:{champion.experiment_id}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except (ProviderError, DownloadError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2
