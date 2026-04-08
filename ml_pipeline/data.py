from __future__ import annotations
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

DEFAULT_EXCHANGE_TZ = "America/New_York"

def get_prices(symbol: str, start: str = "2005-01-01", auto_adjust: bool = True) -> pd.DataFrame:
    df = yf.download(symbol, start=start, auto_adjust=auto_adjust)

    # exchange timezone (fallback to NYSE)
    try:
        tz_name = yf.Ticker(symbol).info.get("exchangeTimezoneName") or DEFAULT_EXCHANGE_TZ
    except Exception:
        tz_name = DEFAULT_EXCHANGE_TZ
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_EXCHANGE_TZ)

    now_local = datetime.now(tz)
    # drop today's incomplete bar before close )
    if not df.empty and df.index[-1].date() == now_local.date() and now_local.time() < dt_time(16, 0):
        df = df.iloc[:-1]

    #multiIndex check 
    is_multi = isinstance(df.columns, pd.MultiIndex) 
    cols = (df.columns.get_level_values(0) #multiIndex fix 
            if is_multi
            else df.columns)
    
    df = (
        df
        .rename_axis(index="trading_date")
        .set_axis(cols.str.lower(), axis=1)
        .assign(symbol=symbol)   
        .reset_index()
        .assign(
                trading_date=lambda x: x["trading_date"].dt.date,
                volume=lambda x: pd.to_numeric(x["volume"], errors="coerce"),
                )
        [["symbol", "trading_date", "open", "high", "low", "close", "volume"]]
        )
    
    return df

def time_series_split(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    if len(y) != n:
        raise ValueError("X and y length mismatch.")
    split = int(n * (1 - val_frac))

    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]