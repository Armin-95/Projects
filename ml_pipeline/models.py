from __future__ import annotations
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import numpy as np


class XGBModel:
    """Scikit-learn-like wrapper."""
    def __init__(self, **params):
        if XGBRegressor is None:    
            raise ImportError("xgboost not installed. `pip install xgboost`")
        default = dict(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            reg_alpha=0.3,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            early_stopping_rounds =30
        )
        default.update(params or {})
        self.model = XGBRegressor(**default)

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.model.fit(X_tr, 
                       y_tr, 
                       eval_set=[(X_val, y_val)],
                        verbose=True
                       )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)
    

class RidgeModel:
    def __init__(self, alphas=None, n_splits: int = 5, **ridge_params):
        if RidgeCV is None:    
            raise ImportError("set up import of RidgeCV or scikit-learn is not installed.")
        
        if alphas is None:
            alphas = np.logspace(2, 5, 40)

        cv = TimeSeriesSplit(n_splits=n_splits)

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas, cv=cv, **ridge_params))
        ])

    def fit(self, X_tr, y_tr):
        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)    

class LSTMModel:
    """Placeholder; implement later with same API (fit/predict/save/load)."""
    def __init__(self, **params):
        raise NotImplementedError("LSTMModel not implemented yet.")
