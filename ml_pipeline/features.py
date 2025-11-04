from __future__ import annotations
import numpy as np
import pandas as pd

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame,symbol):
    df = df.copy()

    # basic indicators / features
    df["RSI14"] = _rsi(df["Close"])
    df["sma5"] = df["Close"].rolling(5).mean()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma5_20_ratio"] = df["sma5"] / df["sma20"]
    df["vol20"] = df["Close"].pct_change().rolling(20).std()
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]

    # volume z-score
    df["vol_mean20"] = df["Volume"].rolling(20).mean()
    df["vol_std20"] = df["Volume"].rolling(20).std()
    df["vol_z"] = (df["Volume",symbol] - df["vol_mean20"]) / df["vol_std20"]

    # calendar / day-of-week
    day_index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    df["day_type"] = day_index.dayofweek

    # target: next-day log return
    df["y_logret_next"] = np.log(df["Close"].shift(-1) / df["Close"])

    df = df.dropna()

    feature_cols = ["RSI14", "day_type", "sma5_20_ratio", "vol20", "range_pct", "vol_z"]
    X = df[feature_cols]
    y = df["y_logret_next"]
    #last_close=  df["Close",symbol].iloc[-1] #train_TEST.py
    #prev_close = df["Close",symbol].iloc[-2] #train_TEST.py
    return X, y, feature_cols #, last_close, prev_close #train_TEST.py
