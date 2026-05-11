from services.ai.ai_client import get_gemini_client
import json
import logging

logger = logging.getLogger(__name__)


def explain_model_comparison_gemini(symbol:str, model_metrics: dict[dict]):
    client = get_gemini_client()
    metrics_text = json.dumps(model_metrics, indent=2, default=str)
    prompt = f"""
        You are explaining machine learning model comparison for a beginner user.

        Stock symbol: {symbol}

        Here are the latest metrics for each model:
        {metrics_text}

        Task:
        Choose which model looks strongest overall based on the metrics.

        Rules:
        - Use simple and easy to understand language.
        - Maximum 4 to 5 sentences.
        - Do not give financial advice.
        - Do not tell the user to buy, sell, or hold the stock.
        - Lower RMSE and MAE are better.
        - Higher hit ratio is usually better.
        - Higher Sharpe ratio is usually better.
        - Higher total return is better.
        - Smaller max drawdown is better.
        - In your decision making, prioritize metrics that are most relevant for most accurate next close price prediction.
        - Highlight the most important metrics that influenced your conclusion with bold text.
        - If one model has better prediction accuracy but worse strategy results, explain that trade-off.
        - End with a clear conclusion about which model looks strongest overall.
        """
    response =client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt)
    
    if response is None:
        logger.error("Gemini response object is None for symbol %s", symbol)
        return None

    if not response.text or not response.text.strip():
        logger.error("Gemini response text is empty for symbol %s", symbol)
        return None
    
    return {"ai_response":response.text.strip(),
             "response_id":response.response_id,
             "ai_model":response.model_version,
             "ai_provider": "google_gemini"}