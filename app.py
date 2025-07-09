from flask import Flask, render_template, request
import yfinance as yf
from functools import lru_cache
import joblib
import os

app = Flask(__name__)

# Load ML model once at startup
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.joblib')
model = joblib.load(MODEL_PATH)

@lru_cache(maxsize=32)
def fetch_data(symbol):
    """Fetch 1 year of history + compute moving averages."""
    df = yf.Ticker(symbol).history(period="1y")
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
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
    closes = df['Close'].tolist()
    volumes = df['Volume'].tolist()
    ma20 = df['MA20'].fillna(method='bfill').tolist()
    ma50 = df['MA50'].fillna(method='bfill').tolist()

    mean_price = round(df['Close'].mean(), 2)
    volatility = round(df['Close'].pct_change().std() * 100, 2)

    return render_template(
        'analysis.html',
        symbol=symbol,
        dates=dates, closes=closes,
        volumes=volumes, ma20=ma20, ma50=ma50,
        mean_price=mean_price,
        volatility=volatility
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Use the trained model to predict next-day close."""
    symbol = request.form['symbol'].upper()
    df = fetch_data(symbol)

    last = df.iloc[-1]
    features = [[
        last['Close'],           # Lag1
        df['Close'].iloc[-2],    # Lag2
        last['MA20'],            # MA20
        last['MA50']             # MA50
    ]]
    prediction = model.predict(features)[0]
    return render_template(
        'predict.html',
        symbol=symbol,
        prediction=round(prediction, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)
