# populate_calendar.py
import exchange_calendars as ecals
from datetime import date
from database.db import init_db, upsert_calendar
import logging
logging.basicConfig(level=logging.INFO)

def main():
    try:
        init_db()
        # Currently all Stock are XNYS (NYSE calendar), if needed CHANGE IN FUTURE!
        cal_code = "XNYS"
        cal = ecals.get_calendar(cal_code)

        start = date(2026, 1, 1)
        end = cal.last_session.date()

        open_days = (cal.schedule.loc[start:end, ["close"]]
                        .assign(calendar_code=cal_code,
                            trading_date=lambda x: x.index.date)
                        .rename(columns={"close": "close_date_time"})
                        .reset_index(drop=True)
                        [["calendar_code","trading_date","close_date_time"]]
                    )

        upsert_calendar(open_days)
        logging.info(f"Inserted {len(open_days)} open days into market_calendar for {cal_code}")

    except Exception:
        logging.exception("Failed to populate market calendar")

if __name__ == "__main__":
    main()