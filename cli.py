import sys
from database.db import init_db, seed_symbols
from experiments.run_experiments import main as run_experiments
import logging
logger = logging.getLogger(__name__)

def main():
    VALID_COMMANDS = {"init_db", "seed", "train"}
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_COMMANDS:
        print("Usage: python cli.py [init_db | seed | train]")
        return

    cmd = sys.argv[1]

    if cmd == "init_db":
        init_db()
        logger.info("DB initialized")

    elif cmd == "seed":
        seed_symbols([
            {"symbol": "AAPL", "calendar_code": "XNYS", "exchange_tz": "America/New_York"},
            {"symbol": "MSFT", "calendar_code": "XNYS", "exchange_tz": "America/New_York"},
            {"symbol": "GOOGL", "calendar_code": "XNYS", "exchange_tz": "America/New_York"},
            {"symbol": "AMZN", "calendar_code": "XNYS", "exchange_tz": "America/New_York"},
            {"symbol": "TSLA", "calendar_code": "XNYS", "exchange_tz": "America/New_York"},
            {"symbol": "META", "calendar_code": "XNYS", "exchange_tz": "America/New_York"},
        ])
        logger.info("Symbols seeded")

    elif cmd == "train":
        run_experiments()
        logger.info("Experiments completed")

if __name__ == "__main__":
    main()