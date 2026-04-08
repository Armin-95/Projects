from datetime import timedelta
import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
#from dotenv import load_dotenv #test env


#load_dotenv()  # Load environment variables from .env file #test env
DATABASE_URL = os.getenv("DATABASE_URL") 
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL")


# SSL check 
if "sslmode" not in DATABASE_URL:
    DATABASE_URL += ("&" if "?" in DATABASE_URL else "?") + "sslmode=require"


def get_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Create tables if not exist."""
    with get_connection() as conn, conn.cursor() as cur:
        # 1) Daily price storage
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_daily_bars_prediction (
            symbol TEXT NOT NULL,
            trading_date DATE NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low  DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            PRIMARY KEY (symbol, trading_date)
        );
        """)

        # 2) stock metadata (timezone + calendar +close time + last update)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_symbols (
            symbol TEXT PRIMARY KEY,
            calendar_code TEXT NOT NULL,
            exchange_tz TEXT NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        # Trading calendar (authoritative open days)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS market_calendar (
            calendar_code TEXT NOT NULL,
            trading_date DATE NOT NULL,
            close_date_time TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (calendar_code, trading_date)
        );
        """)


        # Index for faster lookup in market_calendar by calendar_code and close_date_time  
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_calendar_code_close_dt
            ON market_calendar(calendar_code, close_date_time);
        """)

        # Predict return of next close symbol (with time of the prediction)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_future_return_predictions (
            symbol TEXT NOT NULL,
            trading_date DATE NOT NULL,
            predicted_return DOUBLE PRECISION,
            time_of_prediction TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, trading_date)
        );
        """)

        conn.commit()


def seed_symbols(rows):
    
    #rows: list of dictionaries:
      #{"symbol":"AAPL","calendar_code":"XNYS","exchange_tz":"America/New_York"}
    with get_connection() as conn, conn.cursor() as cur:
        for r in rows:
            cur.execute("""
            INSERT INTO stock_symbols(symbol, calendar_code, exchange_tz)
            VALUES (%s,%s,%s)
            ON CONFLICT(symbol) DO UPDATE SET
              calendar_code=EXCLUDED.calendar_code,
              exchange_tz=EXCLUDED.exchange_tz,
              updated_at = NOW();
            """, (r["symbol"], r["calendar_code"], r["exchange_tz"]))

        conn.commit()

def upsert_calendar(open_days:pd.DataFrame):
    """Open trading days for a calendar_code, with close_date_time.
        After conversion to tuples for ex.: ('2026-03-18 20:00:00+00:00')
    """
    rows = list(open_days.itertuples(index=False, name=None))
    with get_connection() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO market_calendar (
                calendar_code,
                trading_date,
                close_date_time
            )
            VALUES %s
            ON CONFLICT (calendar_code, trading_date)
            DO UPDATE SET
                close_date_time = EXCLUDED.close_date_time
        """, rows)

        conn.commit()


def get_latest_available_close_datetime(symbol: str, datetime_utc_now):
    buffered_utc_now = datetime_utc_now - timedelta(minutes=10)  #safety 10min buffer for close time, it will be comapred to table market_calendar -> close_date_time 
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(""" 
            WITH filtered AS (
                SELECT mc.close_date_time
                FROM market_calendar mc
                JOIN stock_symbols ss
                  ON mc.calendar_code = ss.calendar_code
                WHERE ss.symbol = %s
                  AND mc.close_date_time <= %s
                ORDER BY mc.close_date_time DESC
            )
            SELECT
                (SELECT close_date_time FROM filtered LIMIT 1) AS latest,
                (SELECT close_date_time FROM filtered OFFSET 24 LIMIT 1) AS older_25
        """, (symbol, buffered_utc_now))

        row = cur.fetchone()
        return row if row else (None, None) 


def get_latest_prediction_trading_date(symbol:str):
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute(""" SELECT MAX(trading_date) 
                    FROM stock_daily_bars_prediction 
                    WHERE symbol = %s
                    """, (symbol,)) 
        row = cur.fetchone()
        return row[0] if row and row[0] else None
    

def upsert_prediction_daily_bars(df: pd.DataFrame, symbol:str, start_date):
    # start_date is date from download window range on YF, older than start_date can be deleted from db, because they are already in db 
    if df is None or df.empty or start_date is None:
        return
    rows = list(df.itertuples(index=False, name=None)) # converted to tuples for ex.: ('AAPL',datetime.date(2025, 12, 18),273.35420232069225,273.374203136776,266.7004552212896,271.935546875,51630700),...
    with get_connection() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO stock_daily_bars_prediction (
                symbol,
                trading_date,
                open,
                high,
                low,
                close,
                volume
            )
            VALUES %s
            ON CONFLICT (symbol, trading_date)
            DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,rows)

        cur.execute("""
            DELETE FROM stock_daily_bars_prediction
            WHERE symbol = %s AND trading_date < %s
        """, (symbol, start_date))
        
        conn.commit()


def upsert_stock_future_return_prediction(symbol: str, trading_date, predicted_return: float):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO stock_future_return_predictions (
                symbol,
                trading_date,
                predicted_return
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol, trading_date)
            DO UPDATE SET
                predicted_return = EXCLUDED.predicted_return,
                time_of_prediction = NOW()
        """, (symbol, trading_date, predicted_return))

        conn.commit()


def get_prediction_daily_bars(symbol:str):
    with get_connection()as conn, conn.cursor() as cur:
        cur.execute(""" SELECT trading_date, open, high, low, close, volume
                    FROM stock_daily_bars_prediction 
                    WHERE symbol = %s
                    ORDER BY trading_date DESC
                    LIMIT 25
                    """, (symbol,)) 
        rows = cur.fetchall()

    if not rows:
        return None

    df = pd.DataFrame(
        rows,
        columns=["trading_date", "open", "high", "low", "close", "volume"]
    )

    return df.sort_values("trading_date").reset_index(drop=True)



