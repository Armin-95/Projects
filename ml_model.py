import os
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib
warnings.filterwarnings("ignore")
np.random.seed(42)


def data_features(symbol):

    df = yf.download(symbol, start="2005-01-01", auto_adjust=True)    
    def rsi(series: pd.Series, period: int = 14):
        """Wilder-style RSI with EWM smoothing."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    #return percentage 
    df["ret1"] = df["Close"].pct_change()
    # RSI(14)
    df["RSI14"] = rsi(df["Close"], 14)
    # Trading day type 1..5 (Mon=1, Fri=5)
    df["day_type"] = df.index.dayofweek + 1  # Monday=0 => 1 .. Friday=4 => 5
    # SMA ratio: (SMA5 / SMA20) - 1  (a momentum/mean-reversion signal)
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["sma5_20_ratio"] = df["SMA5"] / df["SMA20"] - 1
    # 20-day volatility (std of daily returns)
    df["vol20"] = df["ret1"].rolling(20).std()
    # Intraday range percentage
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    # Volume z-score (20d)
    df["vol_mean20"] = df["Volume"].rolling(20).mean()
    df["vol_std20"] = df["Volume"].rolling(20).std()
    df["vol_z"] = (df["Volume",symbol] - df["vol_mean20"]) / df["vol_std20"]
    # Target: next-day log-return (shift -1) 
    df["y_logret_next"] = np.log(df["Close"].shift(-1) / df["Close"])
    # Drop warm-up NaNs
    df = df.dropna()
    # + new others features 
    
    feature_cols = [
    "RSI14",
    "day_type",
    "sma5_20_ratio",
    "vol20",
    "range_pct",
    "vol_z"
    # + new others features 
    ]

    X = df[feature_cols]
    y = df["y_logret_next"]

    return X, y 
    
def train_and_save(symbol, algo_model, models_dir='models'):
    
    X, y= data_features(symbol)
    X_train = X
    y_train = y

    # Time-aware internal validation (last 20% of training)
    val_size = max(20, int(0.2 * len(X_train)))
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42
        #early_stopping_rounds =50,
        #save_best=True
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)]
        )  
    
    # ensure directory exists
    os.makedirs(models_dir, exist_ok=True)

    # dynamic save path 
    path = os.path.join(models_dir, f"{algo_model}_{symbol}.joblib")
    joblib.dump(model, path)
    print(f"[{symbol}] Model saved to {path}")
    
if __name__ == '__main__':
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
    algo_model= "xgboost"
    for t in symbols:
        train_and_save(t, algo_model)

    

