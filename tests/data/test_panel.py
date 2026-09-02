import pandas as pd
import pytest
from stockbot.data.panel import build_panel

def _frame():
    return pd.DataFrame({"symbol":["BBB","AAA","BBB","AAA"],"timestamp":pd.to_datetime(["2026-01-02","2026-01-01","2026-01-01","2026-01-02"],utc=True),"open":[2,1,1,2],"high":[2,1,1,2],"low":[2,1,1,2],"close":[2,1,1,2],"volume":[100,100,100,100]})

def test_build_panel_is_canonical_regardless_of_input_order():
    a=build_panel(_frame()); b=build_panel(_frame().sample(frac=1.0,random_state=7).reset_index(drop=True)); pd.testing.assert_frame_equal(a,b); assert a.index.names==["timestamp","symbol"]

def test_build_panel_rejects_duplicate_observations():
    with pytest.raises(ValueError): build_panel(pd.concat([_frame(),_frame().iloc[[0]]],ignore_index=True))
