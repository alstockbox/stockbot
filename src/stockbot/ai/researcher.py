from __future__ import annotations

from typing import Protocol

from stockbot.learning.daily_review import fallback_hypotheses
from stockbot.llm.schemas import ResearchHypothesis


class LLMProvider(Protocol):
    def generate_json(self, context: dict) -> dict | list:
        ...


class ResearchScientist:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    def review(self, context: dict) -> list[ResearchHypothesis]:
        if self.provider is None:
            return fallback_hypotheses(context)

        raw = self.provider.generate_json(context)
        if isinstance(raw, dict):
            items = raw.get("hypotheses")
        else:
            items = raw
        if not isinstance(items, list):
            raise ValueError("LLM provider must return a list of structured hypotheses")
        return [ResearchHypothesis.model_validate(item) for item in items]
