#property strict
#property version   "1.10"
#property description "Read-only StockBot snapshot exporter. Contains no order functions."

input string          InpSymbols = "EURUSD,GBPUSD,XAUUSD";
input string          InpOutputFile = "stockbot_snapshot.json";
input int             InpIntervalSeconds = 1;
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M5;
input int             InpClosedBars = 250;

string JsonEscape(string value)
{
   StringReplace(value, "\\", "\\\\");
   StringReplace(value, "\"", "\\\"");
   return value;
}

string IsoUtc(datetime value)
{
   MqlDateTime dt;
   TimeToStruct(value, dt);
   return StringFormat(
      "%04d-%02d-%02dT%02d:%02d:%02d+00:00",
      dt.year,
      dt.mon,
      dt.day,
      dt.hour,
      dt.min,
      dt.sec
   );
}

string CleanSymbol(string value)
{
   StringTrimLeft(value);
   StringTrimRight(value);
   return value;
}

string BoolJson(bool value)
{
   return value ? "true" : "false";
}

void AppendClosedBars(
   string &json,
   const string symbol,
   const long server_offset
)
{
   json += "\"bars\":[";

   MqlRates rates[];
   int copied = CopyRates(symbol, InpTimeframe, 1, InpClosedBars, rates);
   if(copied > 0)
   {
      for(int i = 0; i < copied; i++)
      {
         if(i > 0)
            json += ",";

         datetime bar_utc = (datetime)((long)rates[i].time - server_offset);
         json += "{";
         json += "\"timestamp\":\"" + IsoUtc(bar_utc) + "\",";
         json += "\"open\":" + DoubleToString(rates[i].open, 12) + ",";
         json += "\"high\":" + DoubleToString(rates[i].high, 12) + ",";
         json += "\"low\":" + DoubleToString(rates[i].low, 12) + ",";
         json += "\"close\":" + DoubleToString(rates[i].close, 12) + ",";
         json += "\"volume\":" + LongToString((long)rates[i].tick_volume);
         json += "}";
      }
   }

   json += "]";
}

void ExportSnapshot()
{
   int handle = FileOpen(
      InpOutputFile,
      FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON
   );
   if(handle == INVALID_HANDLE)
   {
      Print("StockBot exporter: FileOpen failed, error=", GetLastError());
      return;
   }

   datetime utc_now = TimeGMT();
   long server_offset = (long)(TimeTradeServer() - TimeGMT());

   string json = "{";
   json += "\"account\":{";
   json += "\"balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 8) + ",";
   json += "\"equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 8) + ",";
   json += "\"free_margin\":" + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 8) + ",";
   json += "\"currency\":\"" + JsonEscape(AccountInfoString(ACCOUNT_CURRENCY)) + "\",";
   json += "\"timestamp\":\"" + IsoUtc(utc_now) + "\"";
   json += "},";

   json += "\"symbols\":{";

   string symbols[];
   int symbol_count = StringSplit(InpSymbols, ',', symbols);
   bool first = true;

   for(int i = 0; i < symbol_count; i++)
   {
      string symbol = CleanSymbol(symbols[i]);
      if(StringLen(symbol) == 0)
         continue;

      if(!SymbolSelect(symbol, true))
      {
         Print("StockBot exporter: unable to select ", symbol);
         continue;
      }

      MqlTick tick;
      if(!SymbolInfoTick(symbol, tick))
      {
         Print("StockBot exporter: no tick for ", symbol);
         continue;
      }

      datetime tick_utc = (datetime)((long)tick.time - server_offset);

      double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
      if(tick_size <= 0.0)
         tick_size = SymbolInfoDouble(symbol, SYMBOL_POINT);

      long trade_mode = SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);

      if(!first)
         json += ",";
      first = false;

      json += "\"" + JsonEscape(symbol) + "\":{";

      json += "\"quote\":{";
      json += "\"bid\":" + DoubleToString(tick.bid, 12) + ",";
      json += "\"ask\":" + DoubleToString(tick.ask, 12) + ",";
      json += "\"timestamp\":\"" + IsoUtc(tick_utc) + "\"";
      json += "},";

      json += "\"spec\":{";
      json += "\"tick_size\":" + DoubleToString(tick_size, 12) + ",";
      json += "\"tick_value\":" + DoubleToString(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE), 12) + ",";
      json += "\"volume_min\":" + DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN), 8) + ",";
      json += "\"volume_max\":" + DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX), 8) + ",";
      json += "\"volume_step\":" + DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP), 8) + ",";
      json += "\"trade_enabled\":" + BoolJson(trade_mode != SYMBOL_TRADE_MODE_DISABLED);
      json += "},";

      AppendClosedBars(json, symbol, server_offset);

      json += "}";
   }

   json += "}";
   json += "}";

   FileWriteString(handle, json);
   FileFlush(handle);
   FileClose(handle);
}

int OnInit()
{
   if(InpIntervalSeconds < 1 || InpClosedBars < 30)
   {
      Print("StockBot exporter: interval must be >=1 and closed bars >=30");
      return INIT_PARAMETERS_INCORRECT;
   }

   EventSetTimer(InpIntervalSeconds);
   ExportSnapshot();
   Print(
      "StockBot read-only exporter started. FILE_COMMON output: ",
      InpOutputFile
   );
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   ExportSnapshot();
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}
