from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .data import get_prices, time_series_split
from .features import build_features
from .models import XGBModel, LSTMModel

# Default artifacts folder = top-level "models/"
DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "models"


# -------- per-model training functions --------
def train_xgboost(
    symbol: str,
    outdir: Path,
    val_frac: float = 0.2,
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

    X_tr, y_tr, X_val, y_val = time_series_split(X, y, val_frac=val_frac)

    model = XGBModel(**model_params).fit(X_tr, y_tr, X_val, y_val)
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