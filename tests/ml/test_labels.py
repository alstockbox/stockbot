import numpy as np
import pandas as pd
from stockbot.data.panel import build_panel
from stockbot.ml.labels import make_panel_labels

def test_panel_labels_compute_forward_returns_with_per_horizon_tail_nans():
    dates=pd.date_range("2026-01-01",periods=6,freq="B",tz="UTC"); rows=[]
    for symbol,values in [("AAA",[100,101,102,103,104,105]),("BBB",[100,99,98,97,96,95])]:
        for dt,close in zip(dates,values): rows.append({"symbol":symbol,"timestamp":dt,"open":close,"high":close,"low":close,"close":close,"volume":1000})
    labels=make_panel_labels(build_panel(pd.DataFrame(rows)),horizons=(1,5)); aaa=labels.xs("AAA",level="symbol")
    assert np.isclose(aaa.iloc[0]["fwd_return_1"],0.01) and np.isclose(aaa.iloc[0]["fwd_return_5"],0.05)
    assert aaa["fwd_return_1"].isna().sum()==1 and aaa["fwd_return_5"].isna().sum()==5

def test_adverse_excursion_uses_only_forward_window():
    dates=pd.date_range("2026-01-01",periods=4,freq="B",tz="UTC"); rows=[{"symbol":"AAA","timestamp":dt,"open":c,"high":c,"low":c,"close":c,"volume":1000} for dt,c in zip(dates,[100,90,110,120])]
    first=make_panel_labels(build_panel(pd.DataFrame(rows)),horizons=(2,)).xs("AAA",level="symbol").iloc[0]; assert np.isclose(first["adverse_excursion_2"],-0.10)
