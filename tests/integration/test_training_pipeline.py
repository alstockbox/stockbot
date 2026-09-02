import numpy as np
import pandas as pd
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.training_pipeline import run_training_research

def _bars():
    rng=np.random.default_rng(123); dates=pd.date_range("2021-01-04",periods=260,freq="B",tz="UTC"); rows=[]
    for j,symbol in enumerate(["AAA","BBB","CCC","DDD","EEE","FFF","GGG","HHH"]):
        factor=0.00025+0.00008*j; cyclical=0.0015*np.sin(np.arange(len(dates))/(12+j)); shocks=factor+cyclical+rng.normal(0,0.009,len(dates)); close=100*np.exp(np.cumsum(shocks))
        for i,dt in enumerate(dates): rows.append({"symbol":symbol,"timestamp":dt,"open":close[i],"high":close[i]*1.008,"low":close[i]*0.992,"close":close[i],"volume":int(rng.integers(500000,5000000))})
    return pd.DataFrame(rows)

def test_training_pipeline_builds_shared_oos_leaderboard_and_blocks_demo_promotion():
    meta=DatasetMetadata(name="synthetic",source="integration",grade=DataGrade.DEMO); models=[ModelConfig("ridge",seed=7),ModelConfig("extra_trees",seed=7),ModelConfig("hist_gb",seed=7)]; run=run_training_research(_bars(),meta,model_configs=models,horizon=5)
    assert len(run.leaderboard)==3 and all(row.oos_coverage>0 for row in run.leaderboard) and len({row.artifact.dataset_fingerprint for row in run.leaderboard})==1 and run.data_grade is DataGrade.DEMO and run.champion_candidate is None
