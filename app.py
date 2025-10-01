from flask import Flask, render_template, request, jsonify
import yfinance as yf
from datetime import date
from collections import OrderedDict
import joblib
import os
import pandas as pd  
import numpy as np



app = Flask(__name__, template_folder="templates", static_folder="static")

# Enable fast dev feedback
app.config["DEBUG"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Load ML model once at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "models"))

COMPANY_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'META': 'Meta Platforms, Inc.'
}

# Load every .joblib in MODELS_DIR once at startup
MODELS = {}
for fname in os.listdir(MODELS_DIR):
    if fname.lower().endswith(".joblib") and "_" in fname:
        model_type, symbol = os.path.splitext(fname)[0].split("_", 1)
        symbol = symbol.upper()
        model_type = model_type.lower()
        MODELS.setdefault(symbol, {})[model_type] = joblib.load(os.path.join(MODELS_DIR, fname))
if not MODELS:
    raise RuntimeError(f"No .joblib models found in {MODELS_DIR}")

# analyse stored data for 6 tickers - fetch_data   s
CACHE = OrderedDict()
CACHE_MAXSIZE = 6

def fetch_data(symbol: str):
    """Return 1y of data with simple daily cache refresh (max size enforced)."""
    symbol = symbol.upper()
    #
    # What is the latest closed trading day?
    recent = yf.download(symbol, period="5d", interval="1d", auto_adjust=True, progress=False)
    if recent.empty:
        raise RuntimeError(f"No data for {symbol}")
    latest_closed = recent.index[-1].date()

    # If cached and fresh, reuse
    if symbol in CACHE:
        df, last_date = CACHE[symbol]
        if last_date == latest_closed:
            # Move to end (most recently used)
            CACHE.move_to_end(symbol)
            return df

    # Otherwise fetch fresh
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    CACHE[symbol] = (df, df.index[-1].date())
    CACHE.move_to_end(symbol)

    # Enforce max size (drop oldest if too big)
    if len(CACHE) > CACHE_MAXSIZE:
        CACHE.popitem(last=False)

    return df

@app.route('/', methods=['GET'])
def index():
    """Show form to enter ticker symbol."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Fetch data, compute stats, and render chart page."""
    symbol = request.form['symbol'].upper()
    df = fetch_data(symbol)

    dates = df.index.strftime('%Y-%m-%d').tolist()
    closes = df['Close',symbol].tolist()
    volumes = df['Volume',symbol].tolist()
    ma20 = df['MA20'].fillna(method='bfill').tolist()
    ma50 = df['MA50'].fillna(method='bfill').tolist()

    mean_price = round(df['Close',symbol].mean(), 2)
    mean_price_month = round(df['Close',symbol].tail(30).mean(), 2)
    volatility = round(df['Close',symbol].pct_change().std() * 100, 2)
    volatility_month = round(df['Close',symbol].tail(30).pct_change().std() * 100, 2)

    return render_template(
        'analysis.html',
        symbol=symbol,
        dates=dates, closes=closes,
        volumes=volumes, ma20=ma20, ma50=ma50,
        mean_price=mean_price, mean_price_month=mean_price_month, 
        volatility=volatility, volatility_month=volatility_month
    )

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol'].upper()
    company_name = COMPANY_NAMES.get(symbol, symbol)
    df = yf.download(symbol, period="3mo", interval="1d", auto_adjust =True)
    
    ##features for prediction
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder-style RSI with EWM smoothing."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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

    # Target: next-day close (shift -1)# --- target as log-return instead of price ---
    df["y_logret_next"] = np.log(df["Close"].shift(-1) / df["Close"])

    ## Features ##
    feature_cols = [
        "RSI14",
        "day_type",
        "sma5_20_ratio",
        "vol20",
        "range_pct",
        "vol_z"
        # + new others features 
    ]

    ## model select and model prediction

    #data for predict.html page
    prices = df['Close',symbol].ffill().values.tolist()
    times = df.index.strftime('%Y-%m-%d').tolist()
    pred_time = 'Prediction'

    # get all models for this ticker
    models_for_symbol = MODELS.get(symbol, {})
    if not models_for_symbol:
        return f"No models for {symbol}. Available tickers: {list(MODELS.keys())}", 400

    X_pred_data = df[feature_cols].tail(1)
    results = {}

    for model_type, model in models_for_symbol.items():
        predict_ret = float(model.predict(X_pred_data)[0])
        predict_close = float(prices[-1]* np.exp(predict_ret))

        results[model_type] = predict_ret, predict_close


    # Append prediction
    prices_with_pred = prices[-29:] + [predict_close]
    times_with_pred = times[-29:]+ [pred_time]

    return render_template(
        'predict.html',
        company_name=company_name,
        symbol=symbol,
        prediction=predict_close,
        times=times_with_pred,
        prices=prices_with_pred,
        results=results,

    )

@app.route('/api/stock_data')
def stock_data():
    symbol = request.args.get('symbol')
    data = yf.download(symbol, period='2d', interval='5m', auto_adjust=True)
    close_series = data[('Close', symbol)]
    prices = close_series.ffill().values.tolist()
    times = data.index.strftime('%Y-%m-%d %H:%M').tolist()
    return jsonify({'times': times, 'prices': prices})



if __name__ == '__main__':
    pass

