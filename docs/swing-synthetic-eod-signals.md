# StockBot Avanza – syntetisk EOD-signalmotor v1

**2026-10-30 – IMPLEMENTERAD OCH TESTAD I ISOLERAD FORSKNINGSGREN.**
Ingen koppling till Avanza, FX/MT5, betaltjänster, realtidsdata eller
riktig forward paper. Inga externa datakällor har godkänts.

## Syfte och avgränsning

Den tidigare paper-rehearsal-modulen krävde manuella köp-/säljbeslut.
`src/stockbot/swing/paper_signal.py` ger nu **automatiska, deterministiska
signalförslag baserade enbart på observerad syntetisk dagsdata**.
Detta är inte en ny eller utvärderad alpha-strategi: den återanvänder
`SwingConfig` och `engine._indicators` från den befintliga
swingmodellen.

- Hämtar hela XSTO-historiken från den signeringsfria men
  **hash-verifierade** `PaperFeedJournal`. Krav: minst 205 sammanhängande
  XSTO-sessioner och identiskt symbolurval; IPO/avnotering i den här
  signalkedjan saknar ännu separat livscykeltillstånd och **stoppas**.
- Alla dagsposter måste vara observerade vid beräkningsklockan.
  Om nästa XSTO-öppning redan passerat är resultatet
  `EXPIRED_SOURCE`, inte ett bakåtdaterat nybeslut.
- Använder SMA20/50/200, breakout/pullback och 14-dagars ATR.
  En försäljning kan föreslås för befintligt innehav vid
  **slutkurs under SMA50**; befintliga försäljningsbehov prioriteras
  före nya köp.
- Föreslagna köp filtreras på de senaste 20 dagarnas
  syntetiska omsättning och volym, befintliga positioner, högst
  fem öppna innehav, högst 20 % av uppskattat eget kapital i
  en aktie och ungefär 0,5 % ATR-baserad risk per hypotetiskt köp.
  Instrument väljs deterministiskt efter relativ styrka mot SMA50,
  med symbol som stabil tie-break.
- Kassa och hypotetisk position värderas mot senaste **observerade**
  slutkurs. Nya köp blockeras efter minst 8 % förlust relativt
  startkapitalet (`INITIAL_CAPITAL_LOSS_KILL_SWITCH`). Detta är
  **inte** samma sak som en exakt intradag- eller topp-till-botten-
  max drawdown-spärr och kan inte garantera lägre verklig risk.
- Priset vid nästa öppning är **okänt vid beslutet**. Storleken
  är därför en *preliminär* slutkursbaserad beräkning; redan
  implementerad `SEKPaperLedger` kan senare stoppa hypotetiska
  avslut om en gap-öppning gör köpet för dyrt.
- Enbart en signal per dagsperiod får journalföras i
  `SyntheticPaperWorkflow`. Automatbeslut är genom
  `expected_source_entry_sha256` och
  `expected_source_session` låst till precis den datakälla
  den beräknades från. Ingen tyst substitution mellan preview
  och journalföring. Identiska upprepade försök är idempotenta.
- Saknade markeringar för befintliga innehav, för kort
  historik, ändrat symbolurval, framtidsdaterade observationer
  och felaktig datalänk blockeras. Alla rapporter flaggar
  `authorized_real_market_data=false`,
  `actual_forward_paper_started=false` och
  `broker_orders_sent=0`.

## Kommandon

Förutsätter **enbart** verkligt tidsstämplade syntetiska testposter
med tidigare dokumenterade `stage_swing_paper_feed.py`.
Ingen schemalagd drift installeras.

Förhandsgranskning, ingen affär eller journaländring:
```sh
python scripts/suggest_swing_paper_signal.py \
  --feed reports/rehearsal-feed.jsonl \
  --intents reports/rehearsal-intents.jsonl \
  --ledger reports/rehearsal-sek-ledger.jsonl
```

Valfri uttrycklig journalföring av **syntetiskt** beslut före nästa
börsöppning (fortfarande ingen mäklarorder):
```sh
python scripts/suggest_swing_paper_signal.py \
  --feed reports/rehearsal-feed.jsonl \
  --intents reports/rehearsal-intents.jsonl \
  --ledger reports/rehearsal-sek-ledger.jsonl \
  --stage-synthetic
```

När nästa dags **syntetiska EOD** har kommit kan
`scripts/rehearse_swing_paper.py --reconcile` rekonstruera
en hypotetisk next-OPEN-fyllning. Verkliga auktioner,
spread, tick size och intradag-stopp simuleras inte.

## Viktig återstående risk

Det finns **ingen aktiverad automatisk scheduler**, inget tillstånd för
extern data, ingen faktisk riskdriven intradag-stoporder och ingen
oberoende verifierad strategiavkastning. `_indicators` i forskningsmotorn
är dessutom en intern modul-API: implementatören måste bevara
versionerade indikatorregler och regressionstester före produktionsbruk.

Skilj tydligt mellan **TESTAT med syntetiska kurser** och
**PRODUKTIONSVERIFIERAT med lagligt användbar, tidsstämplad data**.
Nästa affärsmässigt viktiga uppgift är licensierat
svenskt historiskt/fortlöpande kursunderlag med
aktieurval, bolagshändelser och kontrollerbara kostnader.
Riktig paper och max 1 000 kr livepilot är separata,
ännu ej godkända steg.
