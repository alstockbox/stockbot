# StockBot V1 Data & Training Architecture

## Mission

V1 ska göra StockBot kapabel att träna och utvärdera flera modeller och strategier på verkliga historiska dataset utan lookahead, random train/test-splits eller orealistiska kostnadsantaganden. Målet är inte att maximera in-sample-avkastning utan att maximera sannolikheten att en upptäckt edge överlever out-of-sample, kostnader, olika marknadsregimer och senare paper trading.

V1 förblir research/backtest/shadow/paper-only. Ingen live-money execution införs.

## Designbeslut

1. **Provider-neutral canonical data layer.** Träningskod får inte bero på en specifik marknadsdataleverantör. Canonical datasets lagras som normaliserade pandas/Parquet-tabeller med tydliga schemas.
2. **Point-in-time före bekvämlighet.** Varje record som kan revideras eller publiceras senare ska kunna bära `event_time` och `available_time`. Beslutsdata filtreras på `available_time <= decision_time`.
3. **Demo-data märks som demo.** En enkel extern prisadapter kan användas för funktionstest, men resultat från data utan survivorship/point-in-time-garantier får inte märkas research-grade.
4. **Cross-sectional training.** V1 ska stödja paneldata över många symboler, inte bara en tidsserie åt gången.
5. **Multiple horizons.** Labels ska stödja exempelvis 1, 5, 20 och 60 handelsdagars forward excess return och adverse excursion, men V1:s standardbenchmark är 5 dagar.
6. **Model zoo bakom ett stabilt interface.** Ridge, ElasticNet, RandomForest/ExtraTrees och HistGradientBoosting ska kunna konkurrera. Modeller får inga specialvägar in i execution.
7. **Time-ordered validation only.** Walk-forward och purged/embargo-splits är obligatoriska för alla trading-performance claims.
8. **Arena väljer på OOS-resultat.** Ranking väger nettoavkastning, excess return, Sharpe/Sortino/Calmar, drawdown, turnover, stabilitet och OOS coverage.
9. **Ingen självmodifierande livebot.** ML/AI får skapa challengers och experiment men champion promotion sker via explicita gates.
10. **Reproducerbarhet.** Dataset fingerprint, feature schema, label config, model config, seed och code/version metadata sparas per experiment.

## Scope

### Ingår i V1

- canonical schemas för bars och point-in-time observations;
- dataset validation och symbol/date filtering;
- panel dataset builder;
- cross-sectional features och ranks;
- multi-horizon labels;
- purged walk-forward splitter med embargo;
- model registry/model zoo;
- deterministic model training;
- out-of-sample prediction assembly;
- experiment result/artifact metadata;
- Alpha Arena för flera model configs;
- leaderboard och champion nomination;
- CLI/demo på lokalt/syntetiskt dataset;
- adapterkontrakt för framtida externa dataproviders;
- optional simple demo price adapter, tydligt markerad non-research-grade.

### Ingår inte i V1

- live orders;
- leverage/short/options;
- RL i produktion;
- automatisk hyperparameter search över enorma sökrymder;
- full nyhets-/fundamentalproviderintegration;
- betald dataabonnemangskoppling;
- GPU-beroende deep learning.

De delarna blir separata milestones efter att denna träningsgrund är verifierad.

## Repository additions

```text
src/stockbot/
  data/
    schemas.py
    validation.py
    panel.py
    providers/
      base.py
      local.py
  features/
    cross_sectional.py
  ml/
    labels.py
    purged_cv.py
    models.py
    trainer.py
    artifacts.py
  arena/
    experiments.py
    leaderboard.py
  research/
    training_pipeline.py
scripts/
  run_training_demo.py
tests/
  data/
  features/
  ml/
  arena/
  integration/
```

## Canonical market data

### Bars

Canonical bar frame använder MultiIndex `(timestamp, symbol)` och minst:

- `open`
- `high`
- `low`
- `close`
- `volume`

Index ska vara sorterat och unikt. Priser måste vara positiva, `high >= max(open, close)`, `low <= min(open, close)`, `high >= low` och volume får inte vara negativ.

### Point-in-time observations

Fundamental/news/macro-liknande framtida data ska följa kontraktet:

- `symbol` eller global scope;
- `event_time`;
- `available_time`;
- `feature_name`;
- `value`;
- optional `source`/`revision_id`.

Feature joins ska använda `available_time`, aldrig framtida slutvärden.

## Dataset quality classification

Varje dataset får quality metadata:

- `DEMO`: funktionstest, inga trading-claims;
- `RESEARCH`: validerad historik med definierade begränsningar;
- `POINT_IN_TIME`: revisions-/availability-säkrad data som uppfyller StockBots researchkrav.

Experimentresultat måste bära dataklass. Arena ska kunna filtrera bort DEMO-resultat från champion promotion.

## Panel feature pipeline

V1 bygger både time-series och cross-sectional features.

### Time-series

Återanvänd V0:s causala features och lägg till:

- 1/5/20/60d returns;
- 20/60/120d momentum;
- realized volatility;
- drawdown från rolling peak;
- volume/liquidity proxies;
- trend distance.

### Cross-sectional

Per timestamp:

- return ranks;
- momentum ranks;
- volatility ranks;
- volume/liquidity ranks;
- sector-neutral hooks för senare providerdata;
- robust percentile/rank transformations.

Cross-sectional features får endast använda symboler som faktiskt finns i samma timestamp snapshot.

## Labels

`make_panel_labels` stödjer:

- forward raw return;
- forward excess return mot benchmark;
- forward cross-sectional rank;
- max adverse excursion under horizon;
- optional binary target `return > cost+hurdle`.

Standard för V1 Arena är 5-day forward excess return. De sista `horizon` observationerna per symbol tas bort.

## Purged / embargo validation

Varje fold består av train/test intervall i kronologisk ordning.

Krav:

- train slutar före test;
- label horizon purgas från träningsslutet;
- explicit embargo mellan train och test;
- ingen random shuffle;
- preprocessing fit endast på train;
- OOS predictions skrivs endast på test rows;
- samma row får aldrig både train- och testroll i samma fold.

V1 implementerar en deterministisk purged walk-forward splitter. CPCV kan läggas ovanpå i nästa milestone när canonical panel-data är etablerad.

## Model zoo

Alla modeller implementerar samma interface:

```python
fit(X, y) -> self
predict_score(X) -> np.ndarray
```

Initiala modeller:

- `ridge`
- `elastic_net`
- `random_forest`
- `extra_trees`
- `hist_gb`

Trädmodeller får begränsad depth/leaf complexity som baseline för att minska överfit. Random seeds är explicita.

## Training pipeline

```text
CANONICAL DATA
  -> VALIDATE
  -> BUILD PANEL FEATURES
  -> BUILD LABELS
  -> CREATE PURGED WALK-FORWARD FOLDS
  -> TRAIN MODEL CONFIGS
  -> OOS PREDICTIONS ONLY
  -> SCORE -> LONG-ONLY SIGNAL
  -> COST/RISK-AWARE BACKTEST
  -> ROBUSTNESS METRICS
  -> ARENA LEADERBOARD
  -> CHAMPION NOMINATION
  -> ARTIFACT METADATA
```

Ingen modell får använda sina train-predictions i performance-rankingen.

## Alpha Arena V1

Arena kör en lista `ExperimentConfig` med:

- model type;
- horizon;
- selected feature set;
- train/test/embargo sizes;
- seed;
- signal scaling;
- hurdle/cost assumptions.

Varje experiment returnerar:

- number of folds;
- OOS sample count/coverage;
- net return;
- benchmark excess return;
- Sharpe/Sortino/Calmar;
- max drawdown;
- turnover/cost;
- directional correlation/information coefficient där lämpligt;
- fold stability;
- research score;
- dataset quality;
- artifact fingerprint.

DEMO-resultat får visas men inte promoveras till research champion.

## Reward / ranking

Arena använder V0:s riskjusterade filosofi men V1 lägger särskild vikt på OOS coverage och fold stability.

Hög rå return ska förlora mot en stabilare modell om den högre returnen drivs av:

- extrem drawdown;
- få OOS samples;
- en enda stark fold;
- hög turnover/cost;
- parameter fragility;
- demo-quality data.

## AI/LLM integration

V1 ändrar inte LLM till ordermotor. Research Scientist får däremot läsa:

- experiment leaderboard;
- fold dispersion;
- feature importance/permutation hooks;
- regime-specific failure metrics;
- cost/turnover diagnostics;
- OOS degradation.

Den får föreslå nästa `ExperimentConfig`, featurefamilj eller validation-test som strukturerad hypothesis. Förslaget måste fortfarande gå genom Arena.

## Error handling

Fail closed på:

- duplicate `(timestamp, symbol)` rows;
- unsorted/invalid timestamps;
- invalid OHLC values;
- non-finite model output;
- train/test overlap;
- insufficient OOS rows;
- DEMO dataset vid promotion request;
- missing benchmark när experiment uttryckligen kräver excess-return label.

## Tests

Minst följande beteenden testas:

1. malformed bars avvisas;
2. duplicate panel rows avvisas;
3. cross-sectional rank använder endast aktuell timestamp;
4. labels droppar framtidsotillgängliga slutrader per symbol;
5. benchmark excess-return labels räknas korrekt;
6. purged folds har train < purge < embargo < test och ingen overlap;
7. preprocessing/model fit sker fold-wise;
8. alla model-zoo modeller ger deterministiska finite predictions;
9. Arena performance baseras endast på OOS predictions;
10. DEMO experiment kan inte promoveras;
11. hög-return/låg-robusthet modell rankas under stabil challenger när penalty motiverar det;
12. end-to-end panel demo producerar leaderboard, artifacts och research hypotheses.

## Success criteria

V1 är färdig när:

- hela V0-testsviten fortfarande är grön;
- alla nya V1-tester är gröna;
- en syntetisk multi-symbol panel kan tränas genom minst tre modellfamiljer;
- endast purged OOS-predictions används i Arena;
- dataset fingerprints/artifact metadata skapas deterministiskt;
- DEMO-data kan demonstrera funktion men blockeras från champion promotion;
- inga live broker/order paths läggs till;
- README dokumenterar exakt skillnaden mellan demo-resultat och research-grade resultat.
