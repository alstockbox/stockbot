import numpy as np
import pandas as pd
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.ml.purged_cv import PurgedWalkForwardSplitter
from stockbot.ml.trainer import train_oos

def test_train_oos_predicts_only_test_rows_and_is_reproducible():
    dates=pd.date_range("2024-01-01",periods=40,freq="B",tz="UTC"); index=pd.MultiIndex.from_product([dates,["AAA","BBB"]],names=["timestamp","symbol"]); x=np.arange(len(index),dtype=float)
    X=pd.DataFrame({"f1":np.sin(x/7),"f2":np.cos(x/11)},index=index); y=pd.Series(0.01*X["f1"]-0.005*X["f2"],index=index,name="fwd_return_5"); splitter=PurgedWalkForwardSplitter(12,4,2,1); meta=DatasetMetadata(name="demo",source="test",grade=DataGrade.DEMO); cfg=ModelConfig(name="ridge",seed=7)
    a=train_oos(X,y,splitter,cfg,meta,label_name="fwd_return_5"); b=train_oos(X,y,splitter,cfg,meta,label_name="fwd_return_5"); pd.testing.assert_series_equal(a.predictions,b.predictions); assert a.predictions.notna().any(); first_train,first_test=next(splitter.split(index)); assert a.predictions.iloc[first_test].notna().all(); assert a.predictions.iloc[first_train].isna().all(); assert 0<a.artifact.oos_coverage<1 and a.artifact.fold_count>0
