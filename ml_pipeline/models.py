from __future__ import annotations
import joblib

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

class XGBModel:
    """Scikit-learn-like wrapper."""
    def __init__(self, **params):
        if XGBRegressor is None:
            raise ImportError("xgboost not installed. `pip install xgboost`")
        default = dict(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            early_stopping_rounds =200,
        )
        default.update(params or {})
        self.model = XGBRegressor(**default)

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

class LSTMModel:
    """Placeholder; implement later with same API (fit/predict/save/load)."""
    def __init__(self, **params):
        raise NotImplementedError("LSTMModel not implemented yet.")
