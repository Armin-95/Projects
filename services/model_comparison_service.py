import hashlib
import json
import logging

from database.ai_db import get_ai_model_comparison_explanation, insert_ai_model_comparison_explanation
from services.ai.model_comparison_ai import explain_model_comparison_gemini
from services.model_metrics_service import _filter_available_metrics, get_all_model_metrics_for_symbol

logger = logging.getLogger(__name__)


def _create_metric_hash(metrics: dict, length: int = 16):
    metrics_json = json.dumps(
        metrics,
        sort_keys=True,
        separators=(",", ":"),
        default=str )
    return hashlib.sha256(metrics_json.encode("utf-8")).hexdigest()[:length]


def get_or_create_ai_model_comparison_explanation(symbol: str, models: dict):
    
    model_metrics = get_all_model_metrics_for_symbol(symbol, models)

    if not model_metrics:
        logger.error(f"No model metrics found for {symbol}. Cannot create AI model comparison explanation.")
        return None
    
    model_metrics_for_ai = _filter_available_metrics(model_metrics)

    metric_hash = _create_metric_hash(model_metrics_for_ai)
    comparison_result = get_ai_model_comparison_explanation(symbol, metric_hash) 

    if comparison_result is None:
        ai_result = explain_model_comparison_gemini(symbol, model_metrics_for_ai)
        if ai_result is None:
            logger.error(f"Failed to get AI explanation from Gemini for {symbol}.")
            return None
        
        explanation = ai_result["ai_response"]
        response_id = ai_result["response_id"]
        ai_model = ai_result["ai_model"]
        ai_provider = ai_result["ai_provider"]

        insert_ai_model_comparison_explanation(symbol, metric_hash, explanation, ai_provider, ai_model, response_id)
        model_comparison_explanation = explanation

    else:
        model_comparison_explanation, ai_model, ai_provider = comparison_result

    return model_comparison_explanation, ai_model, ai_provider