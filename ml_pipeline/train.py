from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from .data import get_prices, time_series_split
from .features import build_features
from .models import XGBModel, LSTMModel
from database.db import insert_model_metrics

# Default artifacts folder = top-level "models/"
DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "models"


# -------- per-model training functions --------
def train_xgboost(
    symbol: str,
    outdir: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15, 
    **model_params,
) -> Dict[str, Any]:
    symbol = symbol.upper()

    print(f"[INFO] Downloading {symbol}")
    df = get_prices(symbol)
    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'.")

    df_features, feature_cols = build_features(df, symbol)
    df_features = df_features.dropna()

    X = df_features[feature_cols]
    y = df_features["y_logret_next"]

    X_tr, y_tr, X_val, y_val , X_test, y_test = time_series_split(X, y, val_frac=val_frac, test_frac=test_frac)

    model = XGBModel(**model_params).fit(X_tr, y_tr, X_val, y_val)

    _calculate_model_metrics(X_test, y_test, symbol, model, model_type="xgboost")

    print(f"[INFO] Training XGBoost (final params={model.model.get_params()})")

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"xgboost_{symbol}.joblib"
    model.save(str(model_path))

    print(f"[SAVED] {model_path}")  
    return {
        "symbol": symbol,
        "model": "xgboost",
        "model_path": str(model_path),
    }


def train_lstm(
    symbol: str,
    outdir: Path,
    val_frac: float = 0.2,
    **model_params,
) -> Dict[str, Any]:
    symbol = symbol.upper()
    raise NotImplementedError("LSTM training isn't implemented yet.")

def _calculate_model_metrics(X_test, y_test,symbol, model, model_type):
    ## model quality check on test set 
    #clarify the variable values 
    y_test_logret = y_test 
    y_test_pred_logret =model.predict(X_test)

    # model metrics on predicted target (log returns)
    mae = mean_absolute_error(y_test_logret, y_test_pred_logret)
    rmse = np.sqrt(mean_squared_error(y_test_logret, y_test_pred_logret)) #sensitive for higher errors(outliers), than MAE
    hit_ratio = (np.sign(y_test_pred_logret) == np.sign(y_test_logret)).mean() #movement of prediction positive/negative same as actual movement of returns
    corrcoef = np.corrcoef(y_test_pred_logret, y_test_logret)[0,1] #does my model catches how well y_test_pred_logret co-move with actual returns-y_test_logret ( direction + magnitude) 

    # model quality based on trading evaluation in simple returns (sr)
    signal = np.sign(y_test_pred_logret)
    y_test_sr= np.exp(y_test_logret) - 1

    strategy_ret = signal * y_test_sr   #trading based on the predictions (long if predicted return positive, short if predicted return negative of NEXT CLOSE) :short/long at the time of CLOSE !
    strategy_mean = strategy_ret.mean() #average return of the strategy per trade
    strategy_std =strategy_ret.std()#volatility of the strategy returns
    sharpe = strategy_mean / strategy_std if strategy_std != 0 else np.nan #risk-adjusted performance, for one unit of risk (std) how much return I get on average (mean), higher -better avg returs per risk unit
    
    max_loss = strategy_ret.min() #max loss per day trade
    cumulative_growth  = (1 + strategy_ret).cumprod() # cumulative return as Series of the strategy over time (as a total growth) 
    total_return = cumulative_growth.iloc[-1] - 1 if not cumulative_growth.empty else np.nan # cumulative return of the strategy over time 

    max_loss = strategy_ret.min() #max loss per day trade
    running_max= cumulative_growth.cummax() #running max of the cumulative return over time
    drawdown = (cumulative_growth - running_max) / running_max
    max_drawdown = drawdown.min()

    #inserting model metrics to DB
    insert_model_metrics(symbol, model_type, mae, rmse, hit_ratio, corrcoef, strategy_mean,
                         strategy_std, sharpe, total_return, max_loss, max_drawdown)


# -------- CLI (single run) --------
def _parse_kv_params(kvs: list[str]) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"Invalid --model_param '{kv}', expected key=value")

        k, v = kv.split("=", 1)

        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass

        extras[k] = v

    return extras


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single model run.")
    parser.add_argument("--symbol", required=True, help="Ticker (e.g., AAPL)")
    parser.add_argument("--model", required=True, choices=["xgboost", "lstm"])
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="Artifacts directory (default: top-level models/)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Validation fraction (0..1)",
    )
    parser.add_argument(
        "--model_param",
        action="append",
        default=[],
        help="Extra model params key=value (repeatable)",
    )
    args = parser.parse_args()

    if not 0 < args.val_frac < 1:
        raise ValueError("--val_frac must be between 0 and 1")

    outdir = Path(args.outdir)
    extras = _parse_kv_params(args.model_param)

    if args.model == "xgboost":
        train_xgboost(args.symbol, outdir, args.val_frac, **extras)
    else:
        train_lstm(args.symbol, outdir, args.val_frac, **extras)


if __name__ == "__main__":
    main()