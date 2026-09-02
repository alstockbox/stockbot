import pandas as pd
from stockbot.ml.purged_cv import PurgedWalkForwardSplitter

def _index(n_dates=24):
    dates=pd.date_range("2026-01-01",periods=n_dates,freq="B",tz="UTC"); return pd.MultiIndex.from_product([dates,["AAA","BBB"]],names=["timestamp","symbol"])

def test_purged_splitter_orders_folds_and_removes_label_overlap():
    index=_index(); splitter=PurgedWalkForwardSplitter(8,3,2,1); splits=list(splitter.split(index)); assert splits; dates=index.get_level_values("timestamp")
    for train_idx,test_idx in splits:
        train_dates=dates[train_idx]; test_dates=dates[test_idx]; unique=pd.Index(dates.unique()); assert train_dates.max()<test_dates.min(); assert unique.get_loc(train_dates.max())<=unique.get_loc(test_dates.min())-3

def test_purged_splitter_applies_embargo_between_test_windows():
    index=_index(30); splitter=PurgedWalkForwardSplitter(8,3,1,2); splits=list(splitter.split(index)); dates=index.get_level_values("timestamp"); all_dates=pd.Index(dates.unique())
    first_end=all_dates.get_loc(pd.Index(dates[splits[0][1]].unique()).max()); second_start=all_dates.get_loc(pd.Index(dates[splits[1][1]].unique()).min()); assert second_start-first_end-1==2
