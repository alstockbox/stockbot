from __future__ import annotations


def combine_signals(signals: dict[str, float], weights: dict[str, float]) -> float:
    if not signals:
        return 0.0
    total = 0.0
    active_weight = 0.0
    for name, signal in signals.items():
        signal = max(0.0, min(1.0, float(signal)))
        weight = max(0.0, float(weights.get(name, 0.0)))
        total += signal * weight
        active_weight += weight
    denominator = max(1.0, active_weight)
    return max(0.0, min(1.0, total / denominator))
