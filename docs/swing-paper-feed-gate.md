# Avanza StockBot — syntetisk paper-feed-mottagning v1

**2026-10-30.** IMPLEMENTERAT för OFFLINE/SYNTHETIC TEST endast.
Inga riktiga börsdata, konton, mäklarorder, betalningar, schemalagda
insamlingar eller forward-paper-portföljer har startats.

## Affärsnytta och motiv

Innan en riktig daglig paper loop får köras måste användning av extern
svensk börsdata vara tillåten, och vi behöver en spårbar källa för vad
boten faktiskt kunde observera **vid den tidpunkt** en signal genererades.
Att backtestet kan räkna med historiska dagsstaplar ger inte detta bevis.

Vår implementation skapar en **separat datainmatningsjournal** som varken
sänder order eller skriver till `SEKPaperLedger`.

`src/stockbot/swing/paper_feed.py`:
- Stöder endast `source_kind=SYNTHETIC_TEST_ONLY`. Externa källor
  nekas explicit med `EXTERNAL_MARKET_DATA_UNLICENSED_OR_UNVERIFIED`.
- Varje mottagen hel XSTO-session använder den **faktiska börskalenderns**
  stängningstid plus 15 minuter. Tidiga stängningar hanteras av kalendern,
  i stället för hårdkodad 17:30-regel.
- Avvisar observationer som daterats efter aktuell körklocka,
  helger/helgdagar, ogiltig OHLC och volym, dubbla symboler och
  saknad/otillåten källreferens.
- Skriver en **atomisk** JSONL-uppdatering med fil-lås, ordnad session,
  sekvens, föregående hash och SHA-256 för kontroll. Efter omstart
  replayverifieras hela filen innan en ny post kan accepteras.
- Accepterar inga ändringar av redan registrerade handelsdagar eller
  retroaktiva sessioner. Journalen är avsedd för append-only-kontroll,
  inte som en autentiserad tredjepartsdatabas.
- `readiness()` rapporterar om börssessioner saknas, om symboluppsättningen
  varit konstant och om minst 205 sessioner finns för teknisk warmup.
  `real_market_data_rights_verified`, `real_forward_paper_allowed`
  och verklig paperaktivitet förblir alltid **false**.
- Hanterar inte bolagshändelser, byten av ISIN, tillstånd till
  varumärkesskyddad marknadsdata eller verklig exekvering. En sådan
  modell kräver separata verifierade datakällor och livscykelregler.

Exempel på **påhittad** indata, inte äkta aktiekurser:
```json
{
  "session": "2025-01-02",
  "observed_at_utc": "2025-01-02T17:00:00+00:00",
  "source_kind": "SYNTHETIC_TEST_ONLY",
  "source_reference": "synthetic:manual_rehearsal",
  "bars": [
    {"symbol": "TEST", "open": 100, "high": 103, "low": 98,
     "close": 101, "volume": 10000}
  ]
}
```

**CLI-test, helt offline:**
```sh
python scripts/stage_swing_paper_feed.py --journal reports/synthetic.jsonl \
  --session-json examples/synthetic_session.json
python scripts/stage_swing_paper_feed.py --journal reports/synthetic.jsonl
```

Notera att dessa paths bara är exempel. Ett kommando skriver lokala
testdata, inget annat.

## Källor och juridisk blockerare

- Nasdaq: https://www.nasdaq.com/market-regulation/nordic/mifid-ii
- Fördröjd handelsdata:
  https://tradereports.nasdaq.com/shares/trade-reports/post-trade
- Kommersiella användningsvillkor:
  https://www.nasdaq.com/docs/data/Nasdaq_European_Data_Policies_September_2026
- Flerårig officiell historik:
  https://www.nasdaq.com/products/data/nordic-baltic/historicalview

Nasdaq beskriver att 15 minuter fördröjd MiFID-data är kostnadsfri
för icke-kommersiell användning, men kommersiella produkter/tjänster och
värdeadderande användning kan kräva avgifter/godkännande.
De nedladdningsbara filerna erbjuds dessutom bara kort tid, inte flera år.
Detta är INTE en licensbekräftelse för Svenska Mjukvarubolaget.

Vår kod laddar inte ner, schemalägger, lagrar eller
återdistribuerar sådan extern data.

## Säkert nästa steg

1. Granska och dokumentera en lagligen användbar historisk källa
   (RAW priser, verkliga avnoteringar, corporate actions, benchmark).
   Ingen köpbeställning eller prenumeration utan uttryckligt godkännande.
2. Utveckla idempotent EOD-signal / nästa-öppningsmodell med tvåstegslogg,
   separat från marknadsdatajournalen och utan mäklarkoppling.
3. Utför ooptimerad utvärdering på oberoende historik, sedan
   framåtriktad observation vid faktisk tidpunkt och tidsstämplade
   paperbeslut innan någon livepilot övervägs.

**Beloppsgräns:** högst 1 000 SEK i en eventuell framtida livepilot,
som fortfarande kräver ett separat uttryckligt ja.
