import logging
import numpy as np
from database.db import get_next_close_date, get_next_close_prediction, upsert_stock_future_prediction
from ml_pipeline.features import get_feature_column


logger = logging.getLogger(__name__)

def get_or_create_next_close_predictions(symbol: str, models_for_symbol: dict, prices: list , df_features):
    results = {}
    try:
        predict_return_trading_date, predict_close_date_time = get_next_close_date(symbol)

        if predict_return_trading_date is None or predict_close_date_time is None:
            logger.error("No future trading_date or close_datetime found for prediction return for symbol %s", symbol)
            return {}

        for model_type, model in models_for_symbol.items():
            predicted_return, predicted_close = get_next_close_prediction(symbol, predict_return_trading_date, model_type)

            if predicted_return is None or predicted_close is None:
                feature_cols = get_feature_column(model_type)
                X_pred_data = df_features[feature_cols].tail(1)
                predicted_return = float(model.predict(X_pred_data)[0])
                predicted_close = float(prices[-1]* np.exp(predicted_return))

                upsert_stock_future_prediction(symbol, model_type, predict_return_trading_date, predict_close_date_time, predicted_return, predicted_close) #new

            results[model_type] = predicted_return, predicted_close

    except Exception:
        logger.exception("get_or_create_next_close_predictions failed for %s", symbol)
        return {}
    

    return results



        