from database.db import get_model_metrics
import logging

logger = logging.getLogger(__name__)



def get_all_model_metrics_for_symbol(symbol: str, models: dict):
    models_for_symbol = models.get(symbol, {})

    if not models_for_symbol:
        logger.error("No models for %s. Available tickers: %s", symbol, list(models.keys()))
        return None
    
    model_metrics ={model_type: get_model_metrics(symbol, model_type)
                    for model_type in models_for_symbol}
    
    if not model_metrics:
        logger.error("No model metrics for symbol: %s and model types: %s", symbol, list(models_for_symbol.keys()))
        return None
    
    return model_metrics

#clear None values for AI explanation
def _filter_available_metrics(model_metrics: dict | None):
    return {
        model_type: metrics
        for model_type, metrics in model_metrics.items()
        if metrics is not None
    }