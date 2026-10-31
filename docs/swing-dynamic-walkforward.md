# Avanza StockBot – dynamiskt XSTO walk-forward med IPO/avnotering

**Status 2026-10-30:** IMPLEMENTERAT i isolerad forskningsgren.
Endast deterministiska syntetiska datakontroller har testats.
**INTE**: godkänd datalicens, verifierad alpha, paper trading eller livehandel.

## Invarianta begränsningar

- Kör först `audit_instrument_history` på hela urvalets samtliga
  XSTO-sessioner: per instrument giltig och sammanhängande noteringstid,
  inga oregistrerade instrument, ingen saknad marknadsdag.
- Varje träningsfold innehåller **exakt** `train_sessions` marknadsdagar;
  testfoldar är separata, och första 200 träningsdagarna används för
  indikatoruppvärmning. Portföljen startar med ny hypotetisk kassa och
  inga positioner på första OOS-testdagen.
- En framtida IPO tas inte med i en tidigare folds instrumentregister.
  En aktie som i det fullständiga historiska materialet senare avnoteras
  är ACTIVE **i tidigare foldar**, utan den framtida slutdagen eller
  avnoteringsmetadata som beslutsunderlag.
- Varje historisk kontantavräkning kontrolleras mot sin sista noterade
  session, betalningsdag och dokumenterade ex ante-beskedstid.
  Settlement efter en fold cutoff är **inte en input** till den foldens
  träning eller test. Historiska IPO/delist-priser fylls aldrig i.
- `CorporateAction` ex-datum filtreras till relevanta scope-sessioner.
  EOD och bolagshändelser kan inte ge upphov till liveorder.
- Varje testfold redovisar `test_lifecycle_events`,
  `train_universe_symbols`, `test_universe_symbols`,
  `train_start`, `train_end` och hypotetiska kostnader/utfall.
- SHA256 över faktisk OHLCV-historik, ISIN-/noteringshistorik,
  kontantavräkningar och bolagshändelser sparas var för sig. En SHA
  verifierar inte juridisk datalicens eller att källan är autentisk.

## CLI-exempel (endast egna, lagligt tillåtna data)

```sh
python scripts/run_swing_research.py \
  --csv data/xsto_raw.csv \
  --manifest data/provenance.json \
  --universe-json data/universe.json \
  --settlements-json data/delisting_cash.json \
  --walkforward --capital-sek 1000 \
  --output reports/dynamic-oos.json
```

Valfria, uttryckligen rättighetsgranskade `--benchmark-csv`
och `--benchmark-manifest` kan ge en **aritmetisk** OOS-jämförelse
mot ett gross-total-return-index (inte bevisad överavkastning).

Standardparametrar: 504 träningstillfällen, 126 OOS-sessioner per fold,
minst 3 foldar. För kort historik, felaktiga livscykler, kontantavräkningar
som saknas, framtidsdaterad information eller företagshändelser som
inte kan hanteras ska testet avvisas.

## Kvarvarande kvalitetsrisker

1. **Överlevnadsbias kan fortfarande finnas.** En avsiktligt ofullständig
   historisk lista kan vara internt konsekvent, och nuvarande auditor
   kan inte bevisa att alla avnoterade bolag finns där.
2. **Endast dokumenterade CASH_ONLY avnoteringar.** Aktiebyteserbjudanden,
   fusioner, likvidation utan verifierat utfall, konkurser med okänd
   recoverability, utdelning i andra instrument och namn-/ISIN-byte
   saknar fullständig ekonomisk behandling.
3. **Dagliga OHLCV är inte börsens orderbok.** Volymgränser och stopfill
   är forskningapproximationer. Spread, tick size, auktion och
   orderkö kan påverka faktisk köpkraft och avslut kraftigt.
4. **Fördröjt kassaflöde, marknadskostnader och skatt** kan ändra netto.
   Redovisade testresultat är hypotetiska, inte ekonomiskt utfall.
5. **Datakällorna är overifierade.** Manifestets källa, datum och villkor
   är självdeklarerade. SHA256 är spårbarhet, inte licensbevis.

Krav innan nästa fas: oberoende godkänd flerårig svensk aktiehistorik
(inklusive även de som senare försvann), bolagshändelser, historiskt
tillåtet totalavkastningsindex, kostnader och benchmark. Därefter
ooptimerad OOS-utvärdering och **riktig, tidsstämplad forward paper**.
Först med separat användargodkännande kan max 1 000 kr livepilot övervägas.
