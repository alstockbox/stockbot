from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator


_FORBIDDEN = re.compile(
    r"\b(execute|place|send)\s+(a\s+)?(live\s+)?(order|trade)\b|\b(buy|sell)\s+[A-Z]{1,6}\s+now\b",
    re.IGNORECASE,
)


class ResearchHypothesis(BaseModel):
    hypothesis: str = Field(min_length=5)
    evidence: list[str] = Field(min_length=1)
    proposed_change: str = Field(min_length=5)
    validation_plan: list[str] = Field(min_length=1)
    invalidation_criteria: list[str] = Field(min_length=1)
    category: str = "research"

    @model_validator(mode="after")
    def prohibit_direct_live_execution(self):
        text = " ".join(
            [
                self.hypothesis,
                self.proposed_change,
                *self.evidence,
                *self.validation_plan,
                *self.invalidation_criteria,
            ]
        )
        if _FORBIDDEN.search(text):
            raise ValueError("research hypotheses may not contain direct live-order instructions")
        return self
