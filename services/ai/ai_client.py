from google import genai
import os
import logging

logger = logging.getLogger(__name__)

_gemini_client = None


def get_gemini_client():
    global _gemini_client

    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    try:
        _gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client created successfully.")
        return _gemini_client

    except Exception:
        logger.exception("Failed to create Gemini client.")
        raise