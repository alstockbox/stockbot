from datetime import datetime, timezone
from stockbot.data.schemas import DataGrade, DatasetMetadata

def test_dataset_metadata_is_explicit_about_data_grade():
    meta = DatasetMetadata(name="synthetic-panel", source="unit-test", grade=DataGrade.DEMO, created_at=datetime(2026, 9, 3, tzinfo=timezone.utc))
    assert meta.grade is DataGrade.DEMO
    assert meta.name == "synthetic-panel"
