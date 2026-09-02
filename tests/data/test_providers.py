import pandas as pd
from stockbot.data.providers.local import LocalFrameProvider
from stockbot.data.schemas import DataGrade, DatasetMetadata

def test_local_provider_filters_symbols_dates_and_preserves_grade():
    bars=pd.DataFrame({"symbol":["AAA","AAA","BBB","BBB"],"timestamp":pd.to_datetime(["2026-01-01","2026-01-03","2026-01-01","2026-01-03"],utc=True),"open":[1,2,3,4],"high":[1,2,3,4],"low":[1,2,3,4],"close":[1,2,3,4],"volume":[100,100,100,100]})
    meta=DatasetMetadata(name="demo",source="test",grade=DataGrade.DEMO); provider=LocalFrameProvider(bars,meta)
    loaded=provider.load_bars(["BBB"],pd.Timestamp("2026-01-02",tz="UTC"),pd.Timestamp("2026-01-04",tz="UTC"))
    assert loaded["symbol"].tolist()==["BBB"] and loaded["close"].tolist()==[4] and provider.metadata.grade is DataGrade.DEMO
