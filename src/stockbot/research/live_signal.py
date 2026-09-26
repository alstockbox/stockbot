from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from stockbot.data.live_models import MarketBar
from stockbot.domain.models import MarketRegime
from stockbot.ensemble.weighted import combine_signals
from stockbot.features.pipeline import build_technical_features
from stockbot.opportunity.ranker import Opportunity
from stockbot.regimes.detector import detect_regime
from stockbot.strategies.baselines import BASELINE_STRATEGIES


REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.BULL_TREND: {
        "trend": 0.35,
        "momentum": 0.35,
        "mean_reversion": 0.08,
        "breakout": 0.17,
    },
    MarketRegime.NEUTRAL_CHOP: {
        "trend": 0.12,
        "momentum": 0.12,
        "mean_reversion": 0.40,
        "breakout": 0.12,
    },
    MarketRegime.BEAR_STRESS: {
        "trend": 0.04,
        "momentum": 0.04,
        "mean_reversion": 0.12,
        "breakout": 0.04,
    },
}


@dataclass(frozen=True)
class LiveSignalConfig:
    min_bars: int = 80
    similarity_band: float = 0.10
    min_edge_samples: int = 12
    shrinkage_samples: float = 20.0
    uncertainty_haircut: float = 0.50

    def __post_init__(self) -> None:
        if self.min_bars < 30:
            raise ValueError("min_bars must be >= 30")
        if not 0.0 < self.similarity_band <= 1.0:
            raise ValueError("similarity_band must be in (0,1]")
        if self.min_edge_samples < 5:
            raise ValueError("min_edge_samples must be >= 5")
        if self.shrinkage_samples < 0.0:
            raise ValueError("shrinkage_samples must be non-negative")
        if self.uncertainty_haircut < 0.0:
            raise ValueError("uncertainty_haircut must be non-negative")


@dataclass(frozen=True)
class LiveSignalAnalysis:
    symbol: str
    regime: MarketRegime
    combined_signal: float
    strategy_signals: dict[str, float]
    expected_return: float
    downside_risk: float
    uncertainty: float
    sample_count: int
    confidence: float
    eligible: bool

    def to_opportunity(self) -> Opportunity | None:
        if not self.eligible or self.expected_return <= 0.0:
            return None
        return Opportunity(
            symbol=self.symbol,
            source="live_baseline_ensemble",
            expected_return=self.expected_return,
            expected_cost=0.0,
            downside_risk=self.downside_risk,
            correlation_penalty=0.0,
            uncertainty=self.uncertainty,
            confidence=self.confidence,
        )


def _bars_frame(bars: list[MarketBar]) -> pd.DataFrame:
    if not bars:
        raise ValueError("bars are required")
    rows = sorted(bars, key=lambda bar: bar.timestamp)
    index = pd.DatetimeIndex([bar.timestamp for bar in rows])
    frame = pd.DataFrame(
        {
            "open": [bar.open for bar in rows],
            "high": [bar.high for bar in rows],
            "low": [bar.low for bar in rows],
            "close": [bar.close for bar in rows],
            "volume": [bar.volume for bar in rows],
        },
        index=index,
    )
    if frame.index.has_duplicates:
        raise ValueError("bar timestamps must be unique")
    return frame


def _signal_row(row: pd.Series) -> tuple[MarketRegime, dict[str, float], float]:
    regime = detect_regime(row)
    strategy_signals = {
        strategy.name: float(strategy.generate_signal(row))
        for strategy in BASELINE_STRATEGIES
    }
    combined = combine_signals(strategy_signals, REGIME_WEIGHTS[regime])
    return regime, strategy_signals, float(combined)


class LiveSignalEngine:
    def __init__(self, config: LiveSignalConfig | None = None) -> None:
        self.config = config or LiveSignalConfig()

    def analyze(self, symbol: str, bars: list[MarketBar]) -> LiveSignalAnalysis:
        if len(bars) < self.config.min_bars:
            raise ValueError(
                f"insufficient bars: need {self.config.min_bars}, got {len(bars)}"
            )

        frame = _bars_frame(bars)
        features = build_technical_features(frame)

        combined = pd.Series(index=features.index, dtype=float)
        regimes: dict[pd.Timestamp, MarketRegime] = {}
        latest_signals: dict[str, float] = {}

        for idx, row in features.iterrows():
            regime, signals, score = _signal_row(row)
            regimes[idx] = regime
            combined.loc[idx] = score
            if idx == features.index[-1]:
                latest_signals = signals

        current_signal = float(combined.iloc[-1])
        current_regime = regimes[features.index[-1]]
        forward_return = frame["close"].pct_change().shift(-1)

        historical_signal = combined.iloc[:-1]
        historical_return = forward_return.iloc[:-1]
        finite = np.isfinite(historical_signal.to_numpy()) & np.isfinite(
            historical_return.to_numpy()
        )
        similar = (
            (historical_signal - current_signal).abs()
            <= self.config.similarity_band
        ).to_numpy()
        mask = finite & similar
        matched = historical_return.iloc[: len(mask)].to_numpy()[mask]
        matched = matched[np.isfinite(matched)]

        sample_count = int(len(matched))
        if sample_count < self.config.min_edge_samples:
            return LiveSignalAnalysis(
                symbol=symbol,
                regime=current_regime,
                combined_signal=current_signal,
                strategy_signals=latest_signals,
                expected_return=0.0,
                downside_risk=0.0,
                uncertainty=1.0,
                sample_count=sample_count,
                confidence=0.0,
                eligible=False,
            )

        mean_return = float(np.mean(matched))
        std_return = float(np.std(matched, ddof=0))
        uncertainty = std_return / math.sqrt(max(1, sample_count))
        shrinkage = sample_count / (
            sample_count + self.config.shrinkage_samples
        )
        conservative_edge = (
            mean_return * shrinkage
            - self.config.uncertainty_haircut * uncertainty
        )
        downside = matched[matched < 0.0]
        downside_risk = (
            float(np.sqrt(np.mean(np.square(downside))))
            if len(downside)
            else 0.0
        )
        confidence = min(1.0, sample_count / 50.0) * max(
            0.0,
            min(1.0, current_signal),
        )
        eligible = conservative_edge > 0.0 and confidence > 0.0

        return LiveSignalAnalysis(
            symbol=symbol,
            regime=current_regime,
            combined_signal=current_signal,
            strategy_signals=latest_signals,
            expected_return=float(max(0.0, conservative_edge)),
            downside_risk=downside_risk,
            uncertainty=float(uncertainty),
            sample_count=sample_count,
            confidence=float(confidence),
            eligible=eligible,
        )
