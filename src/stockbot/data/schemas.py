from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class DataGrade(str, Enum):
    DEMO = "demo"
    BOOTSTRAP = "bootstrap"
    RESEARCH_GRADE = "research_grade"


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    source: str
    grade: DataGrade
    version: str = "1"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
