# StockBot Core Architecture Design

## Mission

StockBot ska byggas som en aggressiv men evidensstyrd research- och tradingplattform vars mål är att maximera långsiktig nettoavkastning och alpha efter courtage, spread och slippage, samtidigt som drawdown, instabilitet och ruinrisk straffas hårt.

Systemet får använda machine learning, LLM:er, statistiska modeller, regelbaserade strategier, ensemblemetoder och senare reinforcement learning när respektive metod kan visa verkligt out-of-sample-värde. Ingen modelltyp är helig; allt konkurrerar på mätbara resultat.

Det finns ingen garanti om perfekt timing eller extrem avkastning. Målet är att skapa den starkaste möjliga processen för att hitta, validera, deploya och kontinuerligt förbättra verklig edge.

## V0 Scope

Första versionen fokuserar på:

- likvida amerikanska aktier och ETF:er;
- long-only;
- ingen leverage;
- inga optioner;
- research, backtesting, live shadow trading och paper execution;
- single-user först;
- broker- och dataprovider-oberoende kärna;
- arkitektur som senare kan utökas med shorting, leverage, options, futures, crypto, svenska aktier och riktiga brokerkonton.

## Grundprinciper

1. **Alpha före aktivitet.** Ingen trade ska tas om förväntad edge inte överstiger kostnader och en definierad hurdle rate.
2. **Riskmotorn är suverän.** Ingen AI-, ML-, LLM- eller strategimodell får åsidosätta hårda riskregler.
3. **Champion / Challenger.** Endast modeller som slår aktuell champion i robusta out-of-sample-tester får ersätta produktionens modell.
4. **Point-in-time-data.** Systemet får aldrig träna eller besluta på information som inte faktiskt var tillgänglig vid beslutstidpunkten.
5. **Samma semantik i backtest och live.** Signaler, sizing, risk och kostnadsmodeller ska dela kontrakt mellan research, paper och senare livehandel.
6. **Regimstyrd ensemble.** Flera specialiststrategier konkurrerar och får olika kapital beroende på marknadsläge och verifierad edge.
7. **Allt versioneras.** Data snapshot, features, prompts, modeller, parametrar, kod-SHA och beslut ska kunna reproduceras.
8. **P&L räcker inte.** Systemet mäter alpha, beta, Sharpe, Sortino, Calmar, max drawdown, turnover, costs, slippage, hit rate, profit factor, expectancy och resultat per marknadsregim.
9. **AI ska förbättra processen, inte få fria händer.** AI får skapa hypoteser och challengers men produktionsändringar kräver validering.

## Hybridarkitektur

StockBot byggs som en egen Python-kärna med tydliga adaptrar runt omkring:

- snabb vektoriserad research för massiv screening;
- separat event-driven simulator för realistisk order- och fill-validering;
- egen ML-, LLM-, ensemble-, portfolio- och risklogik;
- optional LEAN-adapter för backtest/execution där det ger värde;
- brokeradaptrar, initialt paper och senare IBKR;
- FastAPI-backend;
- mobilvänlig webbapp i senare milestone.

## Repository Layout

```text
stockbot/
  apps/
    api/
    web/
  src/stockbot/
    config/
    domain/
    data/
    features/
    regimes/
    strategies/
    ml/
    llm/
    ai/
    ensemble/
    portfolio/
    risk/
    costs/
    backtest/
    evaluation/
    learning/
    execution/
    monitoring/
    arena/
  tests/
  configs/
  scripts/
  reports/
  docs/
```

## Datamotor

### Market data

Systemet ska kunna lagra OHLCV, quotes och senare orderbook/trade-data. Varje observation får både event timestamp och availability timestamp.

### Fundamenta

Rapport- och fundamentaldata måste vara point-in-time. Restatements får inte retroaktivt skapa framtidsinformation i gamla backtests.

### News / Events

Nyheter sparas som timestampade events med källa, berörda instrument, eventtyp, novelty, riktning, osäkerhet och tids-horizon. LLM-bearbetning producerar strukturerade features, aldrig fria BUY/SELL-kommandon.

### Macro / Cross-asset

Arkitekturen stöder räntor, inflation, arbetsmarknad, yield curve, kreditspreadar, FX, råvaror, volatilitetsmått, centralbanksbesked och andra relevanta marknadsvariabler.

## Feature Engine

Varje feature deklarerar:

- inputdata;
- lookback;
- warmup;
- availability/cutoff-regler;
- normalization;
- missing-data-policy;
- version.

Featurefamiljer inkluderar trend, momentum, mean reversion, volatility, liquidity, volume, breadth, correlation, cross-sectional rank, factor exposure, valuation, quality, growth, revisions, earnings surprise, event features, news sentiment, topic intensity och regimfeatures.

## Strategy Arena

Systemet ska från början stödja flera oberoende strategi-familjer:

1. trend / time-series momentum;
2. cross-sectional momentum;
3. relative strength;
4. mean reversion;
5. breakout / volatility expansion;
6. quality + momentum;
7. earnings / event momentum;
8. breadth / risk-on-risk-off;
9. supervised ML ranking;
10. news/event-strategier;
11. senare statistical arbitrage;
12. senare RL-baserade challengers.

Strategier returnerar `Signal`-objekt, inte orders.

## Machine Learning

ML är en kärnkomponent.

### Första modellfamiljer

- regularized linear/logistic baselines;
- gradient boosted trees;
- random forest / extra trees;
- probabilistiska modeller;
- clustering / regime models;
- kalibrerade regressions- och klassificeringsmodeller;
- rankingmodeller;
- senare sequence models;
- senare reinforcement learning efter att enklare modeller etablerat en stark baseline.

### Möjliga targets

- framtida excess return;
- sannolikhet att return överstiger kostnader + hurdle;
- cross-sectional return rank;
- sannolikhet för stor adverse excursion;
- volatilitet / likviditet;
- regime transition probability.

### Validation

- purged time-series cross-validation;
- embargo där labels överlappar;
- walk-forward validation;
- feature selection endast inne i train folds;
- preprocessing fit endast på train-data;
- separat kalibreringsmätning;
- realistiska kostnader;
- out-of-sample perioder;
- multiple-testing / parameter-search-penalty;
- regime splits;
- stress tests;
- Monte Carlo / bootstrap där relevant.

Varje artifact sparar dataset-hash, feature-version, code SHA, parametrar, seed och metrics.

## LLM Layer

LLM ska användas där språkförståelse ger edge, inte som direkt ordermotor.

LLM har fyra huvuduppgifter:

1. tolka nyheter, rapporter, filings och makrohändelser;
2. skapa strukturerade marknadskontext-features;
3. generera nya testbara strategihypoteser;
4. analysera trades och missade möjligheter efteråt.

Alla outputs ska följa versionerade strukturerade schemas med promptversion, modellversion, timestamps och källreferenser.

## AI Research Scientist

Ett särskilt AI-lager ska agera autonom research scientist inom säkra ramar.

Varje dygn ska det kunna:

1. analysera dagens och senaste periodens performance;
2. separera signalproblem från regime-, sizing-, execution-, cost- och dataproblem;
3. leta efter systematiska failure modes;
4. skapa nya feature-, strategy-, sizing- eller riskhypoteser;
5. konfigurera challenger-experiment;
6. läsa experimentresultat;
7. förkasta svaga idéer;
8. nominera starka challengers för vidare validering.

AI:n får inte automatiskt promovera en modell till live champion utan Promotion Gate.

## Regime Engine

Regime Engine uppskattar sannolikheter för bland annat:

- bullish/bearish trend;
- trend strength;
- high/low volatility;
- expansion/contraction;
- liquidity stress;
- cross-sectional dispersion;
- correlation concentration;
- risk-on/risk-off.

En strategi kan få högre/lägre vikt eller stängas av helt i regimer där dess historiska out-of-sample expectancy är svag.

## Ensemble / Meta Learner

Ensemblelagret väger strategier utifrån:

- kalibrerad expected return;
- regime fit;
- rolling out-of-sample quality;
- signal confidence;
- turnover/cost penalty;
- correlation penalty;
- drawdown penalty;
- strategy caps.

En enkel transparent baseline används först. Mer avancerade meta-ML-modeller måste slå denna som challengers innan de får bli champion.

## Reward / Objective Function

Systemet ska aldrig optimera enbart rå P&L.

Research reward ska kombinera exempelvis:

- net return after costs;
- alpha / excess return;
- Sharpe / Sortino;
- Calmar;
- stability across folds/regimes;
- calibration;

med negativa komponenter för:

- max drawdown;
- tail loss / CVaR;
- turnover;
- slippage;
- instability;
- concentration;
- excessive leverage när leverage senare aktiveras;
- probability of ruin;
- parameter fragility;
- multiple-testing burden.

Vikterna i reward-funktionen versioneras och får själva behandlas som en researchfråga.

## Portfolio Construction

Portfolio Engine omvandlar godkända signals till target positions baserat på:

- expected return after costs;
- forecast uncertainty;
- volatility scaling;
- correlation;
- sector/factor exposure;
- liquidity;
- turnover;
- cash buffer;
- risk budget.

Sizing behandlas som en separat alpha/risk-komponent och testas oberoende från signalmodellen.

## Risk Engine

Riskmotorn ligger efter portfolio construction och före execution.

Initiala hårda kontroller:

- max position size;
- max sector/factor concentration;
- max gross exposure;
- max daily turnover;
- liquidity / spread filters;
- stale-data rejection;
- maximum portfolio drawdown state machine;
- daily loss circuit breaker;
- abnormal volatility circuit breaker;
- execution-error circuit breaker;
- kill switch;
- duplicate/conflicting order protection.

Riskmotorn ska kunna modifiera eller neka orderintentioner och alltid returnera reason codes.

## Backtesting Pipeline

### Stage A — Fast Screening

Vektoriserad testmotor används för att testa stora mängder features, parametrar och strategier snabbt.

### Stage B — Robust Validation

Lovande kandidater genomgår purged walk-forward / CPCV-liknande tidsserievalidation, cost sensitivity, regime tests och parameter robustness.

### Stage C — Event-driven Replay

Finalister körs i en event-driven simulator med realistiska ordertider, fill assumptions, partial fills där relevant, slippage, commissions och portfolio constraints.

### Stage D — Shadow / Paper

Kandidaten kör live data utan kapital. Därefter paper-broker execution. Endast kandidater som fortsätter prestera får övervägas för riktig kapitaldeployment.

## Champion / Challenger Promotion Gate

En challenger får endast promoveras när den uppfyller definierade trösklar för:

- out-of-sample net performance;
- risk-adjusted performance;
- max drawdown;
- regime robustness;
- parameter robustness;
- cost sensitivity;
- statistical confidence;
- paper/live-simulation behavior;
- absence of leakage/data-quality violations.

Promotionbeslut sparas som auditerbara artifacts.

## Continuous Learning Loop

Daglig loop:

```text
INGEST
  -> VALIDATE DATA
  -> UPDATE FEATURES
  -> UPDATE REGIME PROBABILITIES
  -> GENERATE SIGNALS
  -> ENSEMBLE
  -> PORTFOLIO CONSTRUCTION
  -> RISK GATE
  -> PAPER/SHADOW EXECUTION
  -> ATTRIBUTION
  -> DRIFT + FAILURE ANALYSIS
  -> AI HYPOTHESIS GENERATION
  -> CHALLENGER EXPERIMENTS
  -> ROBUSTNESS VALIDATION
  -> PROMOTION CANDIDATES
```

Den dagliga kärnfrågan är:

> Vilka verifierbara förändringar kan öka framtida riskjusterad nettoavkastning utan att samtidigt öka drawdown, fragility eller ruinrisk oproportionerligt?

Detta ersätter hindsight-frågan "hur kunde vi tjänat mer idag?".

## Arena

StockBot Arena kör flera bots/modeller parallellt på identiska data och kapitalregler.

Leaderboard visar minst:

- net return;
- benchmark excess return;
- alpha/beta;
- Sharpe;
- Sortino;
- Calmar;
- max drawdown;
- profit factor;
- expectancy;
- hit rate;
- turnover;
- slippage/costs;
- performance by regime;
- sample size;
- model age / live-shadow age.

Ranking ska inte baseras enbart på total return.

## Execution

Execution abstraheras genom ett broker-interface.

Första implementationen är paper execution. Senare läggs IBKR-adapter till utan att strategier behöver ändras.

Orders ska stödja idempotency, status transitions, rejected/partial/filled states och audit trail.

## Observability

Systemet ska logga:

- data freshness;
- feature pipeline health;
- model drift;
- strategy degradation;
- rejected risk decisions;
- execution slippage;
- P&L attribution;
- exceptions;
- experiment lineage.

## Säkerhet och Secrets

API-nycklar, broker credentials och provider credentials får aldrig committas. `.env`, model binaries, local datasets och report artifacts ska ignoreras som standard.

## Definition of Done — Core V0

Core V0 är färdig när projektet kan:

1. ingestera historiska prisdata genom ett provider-interface;
2. validera point-in-time observations;
3. skapa versionerade features;
4. köra minst tre oberoende basstrategier;
5. träna minst en supervised ML challenger;
6. skapa strukturerade LLM/event features via ett provider-interface med deterministic test double;
7. detektera marknadsregim;
8. kombinera signals i en ensemble;
9. bygga target portfolio;
10. köra hard risk gate;
11. köra realistiskt backtest med fees/slippage;
12. producera benchmark- och riskmetrics;
13. registrera experiments;
14. köra champion/challenger promotion checks;
15. köra paper execution;
16. generera daglig research/attribution report;
17. ha automatiserade tester för leakage, costs, risk och promotion logic.

## Explicit Non-Goals for V0

- garanterad avkastning;
- real-money auto trading;
- leverage;
- options;
- 0DTE;
- high-frequency trading;
- egen custody/wallet;
- multi-user SaaS;
- LLM som direkt skickar brokerorders.

Dessa kan läggas till först när kärnan bevisat robust edge och rätt infrastruktur finns.
