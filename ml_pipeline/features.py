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

    ## basic indicators / features

    # 1-day log return
    df["ret1_log"] = np.log(df["close"] / df["close"].shift(1))

    # RSI(14)
    df["rsi14"] = _rsi(df["close"])

    # SMA ratio: (SMA5 / SMA20) - 1  (a momentum/mean-reversion signal)
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma5_20_ratio"] = df["sma5"] / df["sma20"]- 1

    # 20-day volatility (std of daily returns)
    df["vol20"] = df["close"].pct_change().rolling(20).std()
    
    # Intraday range percentage
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    # volume z-score
    df["vol_mean20"] = df["volume"].rolling(20).mean()
    df["vol_std20"] = df["volume"].rolling(20).std()
    df["vol_z"] = (df["volume"] - df["vol_mean20"]) / df["vol_std20"]

    # calendar / day-of-week
    df["day_type"] = pd.to_datetime(df["trading_date"]).dt.weekday

    # target: next-day log return
    df["y_logret_next"] = np.log(df["close"].shift(-1) / df["close"])

    feature_cols = ["ret1_log", "rsi14", "day_type", "sma5_20_ratio", "vol20", "range_pct", "vol_z"]

    return df, feature_cols