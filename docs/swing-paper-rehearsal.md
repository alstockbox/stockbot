# Avanza StockBot – syntetisk EOD → nästa öppning (rehearsal v1)

**Datum:** 2026-10-30. **Kodstatus:** Research implementerad; verklig
forward paper och livehandel är **inte startade**.

## Vad fungerar

Vi har nu tre isolerade, brokerfria journaler:
1. **Feed** (`PaperFeedJournal`): redan färdigställda syntetiska
   handelsdagsstaplar. Varje post sparar källreferens, faktisk
   Nasdaq Stockholm-stängning + 15 minuter och kontrollhash.
2. **Beslut** (`SyntheticPaperWorkflow.plan`): exakt ett syntetiskt
   beslut per observerad handelsdag. Spara (symbol, sida, antal,
   uppskattad avgiftsmodell), exakt källhash och besluts-UTC,
   **innan nästa XSTO-session öppnar**. XSTO-kalendern tar hänsyn
   till helgdagar och helger. Samma beslut kan skickas igen efter
   omstart utan ny post; motstridiga beslut stoppas. En SELL utan
   innehav stoppas redan vid planering.
3. **Hypotetisk bokföring** (`SEKPaperLedger`): först när nästa dags
   **färdigställda** OHLCV har registrerats läser motorn dess OPEN
   och beräknar en efterhandsfyllning, samt drar en **antagen**
   courtageavgift. Kassan måste räcka och blankning tillåts inte.
   Samma `decision_id` bokförs bara en gång. Vid krasch mellan
   bokföring och svar räcker en ny reconcile; befintlig verifierad
   journalpost hittas i stället för att dubblera handeln.

`SyntheticPaperWorkflow` använder samma övergripande fil-lås för
beslutsjournal och reconcile. Feed och SEK-journalen har egna lås.
Syntetiska öppningspriser med nollvolym eller saknat instrument får
ingen hypotetisk fyllning. Ofullständig XSTO-historik stoppas.

**Viktigt:** Nästa dags öppningskurs är inte känd vid öppningen från
en EOD-fil. Vi rekonstruerar därför hypotetiskt pris **efter dagens
stängning**, och registrerar den faktiska observationstidpunkten på
PaperTrade. Det är inte ett verifierat orderavslut på öppningsauktionen,
inte en genomförd order och **inte riktig forward paper trading**.
Modellnamnet är `SYNTHETIC_RETROSPECTIVE_NEXT_XSTO_OPEN`.

Syntetisk avgiftsmodell används endast för reproducerbara tester:
0,5 % av hypotetiskt affärsbelopp, minst 3 SEK per affär. Detta
är **inte** verifierat Avanza-courtage eller faktisk spread/slippage.

## Kommandorad (endast syntetiska data!)

Börja med en tidigare skapad, legal och uttryckligen *syntetisk*
EOD-jsonfil i formatet dokumenterat i `swing-paper-feed-gate.md`.

```sh
python scripts/stage_swing_paper_feed.py \
  --journal reports/rehearsal-feed.jsonl \
  --session-json examples/synthetic-observed-day.json
```

Om `synthetic-observed-day.json` är dagens verkligt tidstämplade
syntetiska testpost **efter stängningen och före nästa öppning**:

```sh
python scripts/rehearse_swing_paper.py \
  --feed reports/rehearsal-feed.jsonl \
  --intents reports/rehearsal-intents.jsonl \
  --ledger reports/rehearsal-sek-ledger.jsonl \
  --plan --symbol TEST --side BUY --quantity 1
```

Efter att **nästa** syntetiska XSTO-session är registrerad med verktyget
ovan kan historisk OPEN räknas fram:

```sh
python scripts/rehearse_swing_paper.py \
  --feed reports/rehearsal-feed.jsonl \
  --intents reports/rehearsal-intents.jsonl \
  --ledger reports/rehearsal-sek-ledger.jsonl \
  --reconcile
```

En andra `--reconcile` återfinner samma journalförda hypotetiska
transaktion; den registrerar inte någon ytterligare.

## Säkerhets- och sanningsgräns

- Ingen Avanza-/Vantage-/MT5-inloggning, token, API, köp, insättning
  eller faktisk börsorder.
- Allt kräver `SYNTHETIC_TEST_ONLY`. Extern marknadsdata har en
  separat rättighetsblockerare. Ingen scraperskapande genväg.
- Det finns ingen automatiskt optimerande eller produktionsaktiv
  stock-selection-/signalagent i denna modul: `--plan` är explicit.
- Courtage, spread, delavslut, prisintervall, börsauktion och
  alternativa corporate actions är ännu ej verifierade mot en
  verklig orderbok eller Avanzas officiella villkor.
- Bruten/tamprad feed/intent/fill hash eller lucka i börssessionerna
  ger ett fail-closed-fel, inte automatisk tyst reparering.
- SHA-256-kedjan ger **ändringsdetektion**, inte autentisk
  signering, leverantörslicens eller ekonomiskt bevis.
- Eventuell riktig forward paper kräver verifierat lagligt använt
  tidsstämplat marknadsflöde **vid beslut och vid öppning**,
  separata noterings-/avnoteringshändelser, korrekt riskkontroll,
  kostnader och verkliga tidsstämplar. Hindsight OPEN är otillräckligt.

**Nästa P0:** rättighetsklarerad svensk aktiedata och ekonomiskt
korrekt exekverings-/kostnadsunderlag. Sedan lagliga riktiga
forward-paper-observationer och senare särskilt godkännande av
eventuell SEK 1 000 livepilot. Ingen automatiskt skapad kostnad.
