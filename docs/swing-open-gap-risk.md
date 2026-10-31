# Avanza-StockBot: öppningsgap och avgiftsinkluderad syntetisk risk

**2026-10-31.** Research/syntetisk simulering, **inte** verklig forward paper,
börsorder, Avanza-konto eller resultatbevis.

## Reproducerbar ändring

Utgångspunkt: en EOD-köpsignal dimensioneras med senast **känd**
slutkurs och ATR. Ett prisgap till nästa öppning kan då göra
den ursprungliga kalkylen fel även om banken hade tillräcklig kassa.
Ingen historisk förbättrad P&L kan påstås genom att använda
nästa dags öppningspris för att optimera *gårdagens signal*.

Den tidigare `SyntheticPaperWorkflow.plan` tar nu emot **valfria**
frysta, före-öppning-kända BUY-riskfält:

- `max_buy_notional_sek` (högsta hypotetiska positionsvärde)
- `max_buy_risk_sek` (högsta planerade stoppförlust **inklusive
  antagna courtage vid både köp och stoppförsäljning**)
- `reference_stop_sek` (EOD-pris minus tidigare känd ATR-stopdistans)

Alla tre krävs tillsammans för automatisk köpbegränsning.
De sparas i samma oföränderliga, SHA-256-länkade
intentjournal som själva beslutet. Frånvarande fält
för äldre/manuella syntetiska beslut behandlas oförändrat;
det är **inte** ett godkännande för riktig börshandel.

### Före framtida handelsdag

`SyntheticSwingSignal` minskar storleken tills
de redan existerande gränserna passeras *inklusive*
uppskattade avgifter på båda sidor av affären.
Ett köp vars avgifter ensamma överskrider riskbudget
ger **NO_SIGNAL**, inte en omöjligt billig mikroaffär.

### Efter avslutad nästa handelsdag (endast syntetisk)

När nästa session är tillgänglig rekonstrueras OPEN i efterhand.
Den ursprungliga BUY-avsikten nekas med exakt status om:

1. `REJECTED_NEXT_OPEN_BELOW_PLANNED_STOP` – öppnade vid/under
   den före öppning planerade stoppreferensen.
2. `REJECTED_NEXT_OPEN_POSITION_LIMIT` – värde vid OPEN över
   den i förväg fixerade positionsgränsen.
3. `REJECTED_NEXT_OPEN_STOP_RISK` – antagen förlust mellan
   OPEN och referensstopp **plus båda antagna avgifterna**
   över förregistrerad stoppriskbudget.

Den ordningen är avsiktlig. Ett avvisat köp skapar **ingen**
`SEKPaperLedger`-transaktion. Vid upprepad avstämning
återfås samma avvisningsstatus; den gamla avvisningen
blockerar inte nya beslut när nästa börsdag är observerad.

Detta modellerar en **öppningsvillkorad begränsningsorder
i en syntetisk efterhandsrekonstruktion**. Den modellerar
**inte** en order som kan fyllas senare under dagen,
och är inte bevis på att en limitorder verkligen skulle
ha fyllts i öppningsauktionen.

## Säkerhetsbegränsningar

- Den avtalade exitnivån **läggs inte hos någon mäklare**.
  En verklig kurs som gapar igenom stoppnivån efter ett
  inträde kan ge större förlust än den beräknade risken.
  Försäljningsorder, intradagstoppar, auktionsköer,
  spread, skatt och skarpa Avanza-avgifter ingår inte.
- Tidiga datum och data från andra leverantörer accepteras
  inte som riktiga börsobservationer. Endast
  `SYNTHETIC_TEST_ONLY` körs. Ingen licens är bevisad.
- Varken tidigare eller nya hypotetiska resultat används
  för att hävda överavkastning mot index.
- `max_drawdown` i nuvarande syntetiska signalmotor
  jämför fortfarande med **ursprungligt kapital** snarare
  än en persistent historical high-water mark. Det är
  en separat kvarstående risk som ska fixas innan
  riktig forward paper tillåts.

## Validering

Nya syntetiska regressioner kontrollerar:
- Normal öppning med efterföljande stor intradagsnedgång
  klassificeras inte i efterhand som *gapavvisning*.
- Var och en av tre typer av gapavvisning; inga fyllningar,
  ingen debiterad kassa; stabilt resultat vid omkörning.
- Äldre journalformat går att läsa; ändringar i riskfält
  utan uppdaterad hashlänk upptäcks.
- Avgifter kan ensamma göra en mikroposition ointressant
  och därmed förhindra journalföring.
- Nya tillåtna separata dagsbeslut blockerar inte
  avvisade tidigare affärer.

Nästa prioritet är verifierad svensk aktiedata med
korrekt licens och en persistent, historisk drawdown-spärr.
Möjlig framtida livepilot på högst SEK 1 000 kräver
användarens separata, uttryckliga godkännande.
