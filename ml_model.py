import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def prepare_features(df):
    df = df.copy()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df = df.dropna()
    X = df[['Lag1','Lag2','MA20','MA50']]
    y = df['Close']
    return X, y

def train_and_save(symbol, path='models/model.joblib'):
    df = yf.Ticker(symbol).history(period="2y")
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == '__main__':
    train_and_save('AAPL')
