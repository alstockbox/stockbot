import numpy as np
import pandas as pd
from stockbot.arena.experiments import ExperimentConfig, run_model_experiment
from stockbot.data.panel import build_panel
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.features.cross_sectional import add_cross_sectional_features
from stockbot.ml.labels import make_panel_labels
from stockbot.ml.models import ModelConfig
from stockbot.ml.purged_cv import PurgedWalkForwardSplitter

def _dataset():
    rng=np.random.default_rng(42); dates=pd.date_range("2022-01-03",periods=180,freq="B",tz="UTC"); rows=[]
    for j,symbol in enumerate(["AAA","BBB","CCC","DDD","EEE","FFF"]):
        shocks=rng.normal(0.0003+j*0.00005,0.01,len(dates)); close=100*np.exp(np.cumsum(shocks)); volume=rng.integers(500000,3000000,len(dates))
        for i,dt in enumerate(dates): rows.append({"symbol":symbol,"timestamp":dt,"open":close[i],"high":close[i]*1.01,"low":close[i]*0.99,"close":close[i],"volume":volume[i]})
    panel=build_panel(pd.DataFrame(rows)); features=add_cross_sectional_features(panel).drop(columns=["open","high","low"],errors="ignore"); labels=make_panel_labels(panel,horizons=(5,))["fwd_return_5"]; return panel,features,labels

def test_model_experiment_returns_only_oos_metrics_and_finite_score():
    panel,features,labels=_dataset(); splitter=PurgedWalkForwardSplitter(60,20,5,5); meta=DatasetMetadata(name="arena",source="test",grade=DataGrade.RESEARCH_GRADE); result=run_model_experiment(panel,features,labels,splitter,ExperimentConfig(ModelConfig("ridge",seed=7)),meta)
    assert result.oos_coverage>0 and np.isfinite(result.score) and np.isfinite(result.metrics["sharpe"]) and result.artifact.label_name=="fwd_return_5"
