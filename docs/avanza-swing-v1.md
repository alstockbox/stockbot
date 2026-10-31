# Avanza Swing — separat aktiehandelsmotor i StockBot

Status 2026-10-29: kodgrenen är under utveckling. Ej svensk marknadsdataverifierad. Ej paper- eller livehandlad.

## Projektgränser
- StockBot: svensk aktie-/ETF-swinghandel, portföljmotor, SEK-redovisning, risk och framtida godkänd Avanza-integration.
- TradBot/MT5/Vantage: separat valuta-/kryptoprojekt. Inga delade konton, kapitalbudgetar, orderflöden eller P&L-böcker.
- SMB Commander/SMB OS får schemalägga forskningsjobb och visa rapporter när tunneln är online. De får inte kringgå handelsriskkontroller.
- Ingen Avanza-inloggning, webbläsarskrapning, BankID-automatisering eller live-orderfunktion implementeras i denna modul.

## Bindande arbetsordning
1. Bygg och testa en korrekt motor med data, signaler, backtesting, risk, portföljbok, testsuite, rapporter och fellägen.
2. Backtesta med juridiskt tillåten historisk svensk data och walk-forward-validering efter alla relevanta kostnader.
3. Kör framåtblickande paper trading i flera marknadslägen. Spara signaler, orderförslag, hypotetiska avslut, avvikelser och resultat.
4. Endast efter godkända resultat, tillåten brokerkoppling och uttryckligt godkännande: starta separat livepilot på MAX 1 000 SEK.
5. Automatiska påfyllningar förbjudna. Varje ökning av kapitalet kräver användarens separata beslut.

## Implementerad researchmodul
Modul: src/stockbot/swing/engine.py.
- Kräver minst 205 synkroniserade dagsstaplar per instrument, stoppar vid ogiltig eller saknad OHLCV-data.
- Långsiktig trend, 20-dagars breakout och trendpullback.
- Dagens stängningssignal -> tidigast nästa handelsdags öppningspris i hypotetisk affär.
- Konservativ ATR-stoppmodell inklusive gap, utan påstående om exakt intradagssekvens.
- Hela aktier, SEK-kassa, maxvikter, planerad risk, courtage, exekveringspåverkan och stoppa-nya-köp vid drawdown.
- Ingen nätverks-/orderadapter. Endast hypotetiska affärer.

Viktiga begränsningar: ingen korrekt börskalender, utdelnings-/splitjustering, handelsstopp, historisk bolagsurvalshistorik, orderbok, delavslut, företagsutdelningar eller skattemodell än. Stoppa-nya-köp är inte ett garanterat maximalt förlustbelopp. Daglig OHLC beskriver inte säker intradagsordning.

## Nästa tester/acceptans
- GitHub CI ska vara grönt inklusive samtliga tidigare projektstester.
- Datakvalitet och point-in-time-stämplar ska verifieras med riktiga svenska instrumentdata.
- Riskgränser ska hårdtestas med gaps, halt, negativa scenarier, dubbla order-ID och kontosaldoförändringar.
- Alla paperavslut ska kunna återskapas i en event journal, även efter omstart.
- Walk-forward OOS, Monte Carlo, omkostnader och benchmarking utvärderas innan forward paper-start.
- Från paper till 1 000-SEK-pilot krävs riskrapport och uttryckligt användargodkännande; kostnader kräver också godkännande.

## Kommando lokalt när Macen fungerar
Installera befintliga projektberoenden och kör hela testsviten: python -m pip install -e '.[dev]' följt av pytest -q.

## Ansvar
Assistent: utveckling, verifiering, källkritik, rapportering. Användare: kontobeslut, finansiering, riskgodkännande, tillåten handel. Mäklare: kontobehörighet, prisdata- och ordervillkor.

## Separat SEK-paperledger

Modul: src/stockbot/swing/paper_ledger.py. Den skriver inga riktiga order och kopplar inte till Avanza.
- Tar emot simulerade orderavslut, sparar beslut-ID, instrument, riktning, antal hela aktier, hypotetiskt pris och avgift i SEK.
- Avvisar dubbla beslut-ID, överköp utan kassa och försäljningar utan ägda aktier.
- Lagrar en kontrollsummebaserad hashkedja, verifierar varje tidigare post vid återstart och låser filen mellan processer.
- Den ursprungliga kassan binds till journalens integritetskontroll. Ett felaktigt ändrat startsaldo stoppar återläsningen.
- Kräver fullständiga aktuella priser för att beräkna equity med öppna positioner.
- Ännu inte ansluten till en live-marknadsscheduler, och paper trading har inte påbörjats.

OBS: befintligt StockBot-shadowlager har USD-fält. SEK-paperledgern återanvänder dess beprövade arkitekturprinciper men får aldrig använda USD-saldon som SEK.


## P0 Datakvalitet och börskalender — 29 oktober 2026

Ny modul: `src/stockbot/swing/data_quality.py`. Använder kostnadsfria
`exchange-calendars` med börskalendern `XSTO` (Stockholm), inte förenklad
måndag–fredag. Importen kräver synkroniserade kompletta handelssessioner och
avvisar börsstängda datum, mellanliggande saknade handelssessioner, ogiltiga
OHLCV-barer, fel valuta och tidsstämplar före färdig dagsstängning.

**Ingen datakälla har verifierats som både kostnadsfri och rättighetsmässigt
lämplig för den avsedda automatiserade handeln.** Nasdaq Nordic erbjuder
historiska produkter men pris och licens behöver godkännas. Yahoo Finance
ger inte allmän rätt till automatiserad handel via dess material; webbaserad
hämtning/scraping ska inte bli produktionsförutsättning.

Prov utan extern uppkoppling:
```bash
python scripts/run_swing_research.py --csv data/bars.csv \
  --manifest data/provenance.json --capital-sek 1000 \
  --output reports/xsto-backtest.json
```

Minimal manifestfil `data/provenance.json`:
```json
{
  "provider": "namn på faktisk datakälla",
  "terms_reference": "länk eller referens till användningsvillkor",
  "collected_at_utc": "2026-10-29T12:00:00Z",
  "currency": "SEK",
  "price_basis": "RAW",
  "corporate_actions_checked": false,
  "historical_universe_checked": false
}
```

Uppge korrekta källuppgifter. Valideraren skapar deterministiska SHA-256-
fingeravtryck för data och manifest, men **kan inte oberoende bekräfta**
datakällans licens, korrekt historiskt aktieuniversum, utdelningar,
splitar eller att riktiga avslut hade varit möjliga. Den lämnar därför
alltid `research_grade=UNVERIFIED_PROVENANCE_AND_ACTIONS` och
`order_execution_permitted=false`. Dataset med handelsstopp,
bolag som nyintroducerats, avnoterade bolag eller bristfällig data kräver
utbyggd händelsemodell i stället för att artificiellt fylla prisrader.

Backtest utan `--manifest` är möjligt för syntetiska enhetstester, men
rapporteras uttryckligen som `NOT_AUDITED`.

Separat Forex/Krypto-bot berörs inte. Handelskonto, inloggning,
orderläggning och kapitalinsättning är fortsatt avstängt.

**Acceptans för nästa steg:** verifierad rätt att lagra/använda svensk
historik, bolagshändelser och relevant totalavkastningsbenchmark, därefter
kostnadskänslig historisk walk-forward innan framåtpapper.


## OOS walk-forward och parameterdisciplin — 29 oktober 2026

Ny modul `src/stockbot/swing/walkforward.py`.
- Standardfönster: **504 handelsdagar träning**, **126 handelsdagar test**,
  minst **3 kompletta testfönster** (882 sessions inklusive första träningen).
- Begränsat, på förhand definierat ATR-stopputrymme: **2,0 / 2,5 / 3,0**.
  Enbart träningsperiodens realiserade hypotetiska nettoresultat minus
  drawdown-straff får styra vilket alternativ som väljs.
- OOS-fönstret startar med **tom portfölj** och ett nytt, hypotetiskt
  saldo. Den får läsa tidigare prisstaplar för indikatorernas uppvärmning
  men får inte låtsas ha öppnat aktier på träningshistoriken.
- OOS-dagar påverkar **inte** den foldens modellval. Testet ändrar
  framtida kursvärden för att kontrollera denna gräns.
- Testperioderna är kronologiska utan överlappning. Hypotetiska
  testresultat sätts samman; friktionskänslighet kontrolleras separat vid
  10 och 30 bps exekveringspåverkan per sida (standard).
- `first_trade_date` blockerar köp före OOS-perioden även om tekniska
  signaler finns i träningshistoriken.
- Resultat = **WALK_FORWARD_BACKTEST_ONLY**, `money_traded_sek=0`,
  `independent_benchmark_validated=false`,
  `sufficient_activity_for_live_decision=false`.

Kör först XSTO-granskad historik, sedan walk-forward:
```bash
python scripts/run_swing_research.py --csv data/bars.csv \
  --manifest data/provenance.json --capital-sek 1000 \
  --walkforward --output reports/xsto-walkforward.json
```

Detta är **inte** bevis på överavkastning eller tillstånd för paper/live.
Avsaknaden av verifierade svenska corporate actions, historiskt aktieuniversum,
databaserad totalavkastningsbenchmark och handelslikviditet kvarstår.

Faktaunderlag att beakta vid datakällsval:
- Nasdaq Nordic HistoricalView: https://www.nasdaq.com/products/data/nordic-baltic/historicalview
- Nasdaq Nordics datautbud: https://www.nasdaq.com/products/data/equities/nordic-baltic
- Yahoo Finance dataanvändning: https://help.yahoo.com/kb/finance/article-exchanges-data-delays-sln2310.html
- XSTO-kalender i exchange_calendars: https://github.com/gerrymanoim/exchange_calendars
