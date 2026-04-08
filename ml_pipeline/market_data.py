from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
from database.db import get_latest_available_close_datetime, get_latest_prediction_trading_date, upsert_prediction_daily_bars
from database.populate_calendar import main as populate_calendar
import logging

logger = logging.getLogger(__name__)

def _get_available_close_datetimes(symbol, utc_now):
    latest_close_dt, older_25_close_dt  = get_latest_available_close_datetime(symbol, utc_now)
    if latest_close_dt is None: # Populate calendar if no data available for this ticker, then re-check available close datetimes after calendar is populated
        populate_calendar()
        latest_close_dt, older_25_close_dt  = get_latest_available_close_datetime(symbol, utc_now)

    return latest_close_dt, older_25_close_dt


def _determine_download_range(latest_close_dt, older_25_close_dt):
    if latest_close_dt is None or older_25_close_dt is None:
        return None, None
    
    start_date = older_25_close_dt.date() # I need only date to set up YF download start date (not datetime)
    end_date = latest_close_dt.date() + timedelta(days=1) #end= in YF download is exclusive, Add 1 day to include latest_close_dt date
    
    return start_date, end_date


def _download_daily_data(symbol, start_date, end_date):
     if start_date is None or end_date is None or start_date >= end_date:
        return None
    
     df = yf.download(symbol, start=start_date ,end=end_date, interval="1d", auto_adjust =True, progress=False)

     return df


def _prepare_daily_data(df,symbol):
    if df is None or df.empty:
        return None
     #prepare data for upsert in db, remove multiIndex if there is 
    is_multi = isinstance(df.columns, pd.MultiIndex) #multiIndex check 
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
    
    return (df)


def sync_prediction_daily_data(symbol):
    try:
        utc_now = datetime.now(timezone.utc)

        latest_close_dt, older_25_close_dt = _get_available_close_datetimes(symbol, utc_now)

        if latest_close_dt is None:
            return False
        
        start_date, end_date = _determine_download_range(latest_close_dt, older_25_close_dt)

        if start_date is None or end_date is None:
            return False

        last_stored_close_date = get_latest_prediction_trading_date(symbol)

        if last_stored_close_date is None:
            raw_df = _download_daily_data(symbol, start_date, end_date)

        elif latest_close_dt.date() > last_stored_close_date: # some last entries of close symbol missing in db, fetch from YF 
            if last_stored_close_date > older_25_close_dt.date(): # when less than 25 entries of close symbol missing in db (complete last 25 available entries)
                start_date = last_stored_close_date + timedelta(days=1) 
            raw_df = _download_daily_data(symbol, start_date, end_date)

        else:
            return False
        
        if raw_df is None or raw_df.empty:
            return False

        prepared_df = _prepare_daily_data(raw_df, symbol)
        if prepared_df is not None and not prepared_df.empty:
            upsert_prediction_daily_bars(prepared_df, symbol, start_date)
            return True
        
        return False
    
    except Exception:
        logger.exception("sync_prediction_daily_data failed for %s", symbol)
        return False
        

     