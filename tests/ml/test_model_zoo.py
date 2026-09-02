import numpy as np
import pytest
from stockbot.ml.models import ModelConfig, build_model

@pytest.mark.parametrize("name",["ridge","elastic_net","extra_trees","random_forest","hist_gb"])
def test_model_zoo_models_are_deterministic_and_finite(name):
    rng=np.random.default_rng(7); X=rng.normal(size=(120,6)); y=0.4*X[:,0]-0.2*X[:,1]+rng.normal(0,0.05,120); cfg=ModelConfig(name=name,seed=11)
    a=build_model(cfg).fit(X,y).predict(X[:12]); b=build_model(cfg).fit(X,y).predict(X[:12]); assert np.all(np.isfinite(a)); np.testing.assert_allclose(a,b)

def test_model_zoo_rejects_unknown_model():
    with pytest.raises(ValueError): build_model(ModelConfig(name="magic_model"))
