"""StockBot's Swedish-equity swing research module.

Research and backtesting only. No Avanza login, scraping or live orders.
"""

from stockbot.swing.engine import Fill, SwingConfig, SwingResult, run_swing_backtest, validate_ohlcv

__all__ = ["Fill", "SwingConfig", "SwingResult", "run_swing_backtest", "validate_ohlcv"]
