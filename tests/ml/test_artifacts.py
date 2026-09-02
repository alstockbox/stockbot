import pandas as pd
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.artifacts import dataset_fingerprint

def test_dataset_fingerprint_changes_with_values_or_grade():
    frame=pd.DataFrame({"x":[1.0,2.0]},index=pd.Index(["a","b"],name="row")); demo=DatasetMetadata(name="x",source="test",grade=DataGrade.DEMO); research=DatasetMetadata(name="x",source="test",grade=DataGrade.RESEARCH_GRADE)
    base=dataset_fingerprint(frame,demo); assert base!=dataset_fingerprint(frame.assign(x=[1.0,3.0]),demo); assert base!=dataset_fingerprint(frame,research); assert base==dataset_fingerprint(frame.copy(),demo)
