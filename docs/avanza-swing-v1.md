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
