# riskradar/utils/market_data.py
import pandas as pd
import logging
from .data_fetcher import get_quote_data # Import from the new centralized fetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_price_and_volatility(tickers: list, api_keys: dict) -> pd.DataFrame:
    """
    Fetches the latest price, change percentage, and basic volatility for a list of tickers.
    Uses the centralized, cached data_fetcher.
    """
    data = []
    logging.info(f"Fetching market snapshot for tickers: {tickers}")

    for ticker in tickers:
        # Call the centralized function which handles fetching, fallbacks, caching, and retries
        quote_result = get_quote_data(ticker, api_keys)
        data.append(quote_result)

    logging.info("Finished fetching market snapshot.")
    return pd.DataFrame(data)

# Removed old fetch_from_finnhub, fetch_from_alphavantage, fetch_from_yahoo, and throttle_wait
# as this logic is now handled within data_fetcher.py