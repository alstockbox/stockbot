import numpy as np
import pandas as pd
from stockbot.data.panel import build_panel
from stockbot.features.cross_sectional import add_cross_sectional_features

def _panel(n=30):
    dates=pd.date_range("2026-01-01",periods=n,freq="B",tz="UTC"); rows=[]
    for symbol,drift in [("AAA",0.02),("BBB",0.0),("CCC",-0.01)]:
        close=100*np.cumprod(np.full(n,1+drift))
        for i,dt in enumerate(dates): rows.append({"symbol":symbol,"timestamp":dt,"open":close[i],"high":close[i],"low":close[i],"close":close[i],"volume":1000+i*(1 if symbol=="AAA" else 2)})
    return build_panel(pd.DataFrame(rows))

def test_cross_sectional_momentum_rank_orders_symbols_on_same_date():
    features=add_cross_sectional_features(_panel()); latest=features.xs(features.index.get_level_values("timestamp").max(),level="timestamp")
    assert latest.loc["AAA","momentum_rank"]>latest.loc["BBB","momentum_rank"]>latest.loc["CCC","momentum_rank"]

def test_cross_sectional_features_do_not_change_past_when_future_is_appended():
    panel=_panel(30); cutoff=panel.index.get_level_values("timestamp").unique()[24]; short=panel.loc[(slice(None,cutoff),slice(None)),:]
    full_features=add_cross_sectional_features(panel); short_features=add_cross_sectional_features(short); pd.testing.assert_frame_equal(full_features.loc[short_features.index],short_features)
