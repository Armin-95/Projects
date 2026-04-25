from __future__ import annotations
#from pathlib import Path
from typing import Any, Dict, List, Tuple
from ml_pipeline.train import train_ridge, train_xgboost, train_lstm, DEFAULT_OUTDIR


# Each item (symbol, model_name, params_dict) add custom params if needed in dictionary
EXPERIMENTS: List[Tuple[str, str, Dict[str, Any]]] = [
    ("AAPL", "xgboost", {}),
    ("MSFT", "xgboost", {}),
    ("GOOGL", "xgboost", {}),
    ("AMZN", "xgboost", {}),
    ("TSLA", "xgboost", {}),
    ("META", "xgboost", {}),
    ("AAPL", "ridge", {}),
    ("MSFT", "ridge", {}),
    ("GOOGL", "ridge", {}),
    ("AMZN", "ridge", {}),
    ("TSLA", "ridge", {}),
    ("META", "ridge", {})
    # ("AAPL", "lstm", {...})  # when LSTM is implemented
    # pamram for exp for Xgboost: {"n_estimators": 5000, "learning_rate": 0.03}, default params are in models.py, override by passing in params dict here
]

OUTDIR = DEFAULT_OUTDIR 

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for i, (symbol, model, params) in enumerate(EXPERIMENTS, start=1):
        try:
            print(f"\n[JOB {i}] {symbol} | {model} -> {OUTDIR}")
            if model == "xgboost":
                result = train_xgboost(symbol, OUTDIR, **params)
            elif model == "ridge":
                result = train_ridge(symbol, OUTDIR, **params)
            elif model == "lstm":
                raise NotImplementedError("LSTM training isn't implemented yet.")
            else: 
                raise ValueError(f"Unknown model '{model}'")
            print(f"[DONE] {result}")
        except Exception as e:
            print(f"[ERROR] {symbol} | {model}: {e}")

if __name__ == "__main__":
    main()
