from dataclasses import replace
from stockbot.arena.leaderboard import eligible_for_promotion, rank_experiments
from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.artifacts import ExperimentArtifact

def _result(name,score,cagr,drawdown,turnover,stability=0.8):
    artifact=ExperimentArtifact("abc",("x",),"target",name,{},7,4,0.6,{}); return ModelExperimentResult(name,score,{"cagr":cagr,"max_drawdown":drawdown,"turnover":turnover},stability,0.6,artifact)

def test_rank_experiments_uses_research_score_not_raw_return():
    assert rank_experiments([_result("reckless",1.0,0.80,0.65,70.0,0.2),_result("stable",3.0,0.25,0.10,4.0)])[0].name=="stable"

def test_demo_data_can_never_be_promotion_eligible():
    result=_result("winner",5.0,0.4,0.08,3.0); demo=DatasetMetadata(name="demo",source="test",grade=DataGrade.DEMO); research=replace(demo,grade=DataGrade.RESEARCH_GRADE); assert eligible_for_promotion(result,demo) is False and eligible_for_promotion(result,research) is True
