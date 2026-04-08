from __future__ import annotations
#from pathlib import Path
from typing import Any, Dict, List, Tuple
from ml_pipeline.train import train_xgboost, train_lstm, DEFAULT_OUTDIR


# Each item (symbol, model_name, params_dict) add custom params if needed in dict

EXPERIMENTS: List[Tuple[str, str, Dict[str, Any]]] = [
    ("AAPL", "xgboost", {}),
    ("MSFT", "xgboost", {}),
    ("GOOGL", "xgboost", {}),
    ("AMZN", "xgboost", {}),
    ("TSLA", "xgboost", {}),
    ("META", "xgboost", {})
    # ("AAPL", "lstm", {...})  # when LSTM is implemented
    # pamram for exp: {"n_estimators": 5000, "learning_rate": 0.03}, default params are in models.py, override by passing in params dict here
]

VAL_FRAC = 0.2
OUTDIR = DEFAULT_OUTDIR 

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for i, (symbol, model, params) in enumerate(EXPERIMENTS, start=1):
        try:
            print(f"\n[JOB {i}] {symbol} | {model} -> {OUTDIR} (val_frac={VAL_FRAC})")
            if model == "xgboost":
                result = train_xgboost(symbol, OUTDIR, VAL_FRAC, **params)
            elif model == "lstm":
                result = train_lstm(symbol, OUTDIR, VAL_FRAC, **params)
            else:
                raise ValueError(f"Unknown model '{model}'")
            print(f"[DONE] {result}")
        except Exception as e:
            print(f"[ERROR] {symbol} | {model}: {e}")

if __name__ == "__main__":
    main()
