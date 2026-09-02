import pandas as pd
import pytest
from stockbot.data.validation import as_of_filter, validate_bars

def _bars():
    return pd.DataFrame({"symbol":["AAA","AAA","BBB","BBB"],"timestamp":pd.to_datetime(["2026-01-01","2026-01-02","2026-01-01","2026-01-02"],utc=True),"open":[10.,11.,20.,21.],"high":[11.,12.,21.,22.],"low":[9.,10.,19.,20.],"close":[10.5,11.5,20.5,21.5],"volume":[1000,1100,2000,2100]})

def test_validate_bars_accepts_sorted_canonical_rows(): validate_bars(_bars())

@pytest.mark.parametrize("mutator",[lambda f:pd.concat([f,f.iloc[[0]]],ignore_index=True),lambda f:f.assign(close=[0.,11.5,20.5,21.5]),lambda f:f.assign(volume=[-1,1100,2000,2100]),lambda f:f.iloc[[1,0,2,3]].reset_index(drop=True)])
def test_validate_bars_rejects_bad_market_data(mutator):
    with pytest.raises(ValueError): validate_bars(mutator(_bars()))

def test_as_of_filter_excludes_information_not_yet_available():
    frame=pd.DataFrame({"value":[1,2],"available_time":pd.to_datetime(["2026-01-02","2026-01-05"],utc=True)})
    assert as_of_filter(frame,pd.Timestamp("2026-01-03",tz="UTC"))["value"].tolist()==[1]
