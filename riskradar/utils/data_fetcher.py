# riskradar/utils/data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import streamlit as st
import logging
import time
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Exception for retry logic
class APIError(Exception):
    pass

class DataFetchError(Exception):
    pass

# --- Tenacity Retry Configuration ---
RETRY_WAIT = wait_random_exponential(multiplier=1, max=30) # Wait 1s, 2s, 4s,... up to 30s + random jitter
RETRY_STOP = stop_after_attempt(4) # Try 4 times in total (1 initial + 3 retries)
RETRY_ON_EXCEPTION = retry_if_exception_type((requests.exceptions.RequestException, APIError, DataFetchError))

# --- API Call Functions with Retry Logic ---

@retry(wait=RETRY_WAIT, stop=RETRY_STOP, retry=RETRY_ON_EXCEPTION)
def _fetch_with_retry(url, params=None, headers=None):
    """Generic function to fetch data from URL with retry logic."""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15) # Added timeout
        if response.status_code == 429:
            logging.warning(f"Rate limit hit (429) for {url}. Retrying...")
            reset_time = response.headers.get("Retry-After") # Check for Retry-After header
            if reset_time:
                try:
                    wait_time = int(reset_time)
                    logging.info(f"Following Retry-After header: waiting {wait_time} seconds.")
                    time.sleep(wait_time + 1) # Add a buffer
                except ValueError:
                    # If Retry-After is not an integer (e.g., a date), use default backoff
                     logging.warning(f"Could not parse Retry-After header value: {reset_time}. Using default backoff.")
                     # Let tenacity handle the backoff based on RETRY_WAIT
            else:
                 # Let tenacity handle the backoff based on RETRY_WAIT
                 pass
            raise APIError(f"Rate limit exceeded (429) for {url}") # Raise error to trigger retry

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        logging.warning(f"Request timed out for {url}. Retrying...")
        raise APIError(f"Request timed out for {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
        raise # Re-raise to be caught by tenacity if it's a connection error etc.
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching {url}: {e}")
        raise # Re-raise other unexpected errors


@st.cache_data(ttl=timedelta(minutes=15)) # Cache results for 15 minutes
def get_historical_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetches historical market data for a given ticker using yfinance.
    Caches the result. Includes basic retry for empty data.
    """
    logging.info(f"Fetching historical data for {ticker} (Period: {period})")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        # Basic retry if data is empty (might indicate temporary issue)
        attempt = 0
        max_attempts = 2
        while hist.empty and attempt < max_attempts:
            attempt += 1
            logging.warning(f"yfinance returned empty history for {ticker}. Attempt {attempt}/{max_attempts} after delay.")
            time.sleep(2 * attempt) # Simple linear backoff for this specific case
            hist = stock.history(period=period)

        if hist.empty:
            logging.error(f"yfinance failed to return history for {ticker} after {max_attempts} attempts.")
            # Return an empty DataFrame with expected columns to prevent downstream errors
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])
            # Or raise a specific error: raise DataFetchError(f"No historical data found for {ticker}")

        return hist
    except Exception as e:
        # Check for specific yfinance errors if possible, otherwise log general error
        logging.error(f"Error fetching yfinance history for {ticker}: {e}")
        # Example check for JSON decode error mentioned in user logs
        if "Expecting value: line 1 column 1" in str(e):
             logging.error(f"Potential JSON decode error for {ticker}. Response might be malformed.")
        # Return empty DataFrame on error
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])
        # Or raise: raise DataFetchError(f"yfinance error for {ticker}: {e}") from e


@st.cache_data(ttl=timedelta(minutes=5)) # Cache quote data for 5 minutes
def get_quote_data(ticker: str, api_keys: dict) -> dict:
    """
    Fetches current quote data (price, change, basic volatility).
    Uses yfinance first, then fallbacks (Finnhub, Alpha Vantage).
    Includes retry logic via _fetch_with_retry for fallbacks.
    """
    logging.info(f"Fetching quote data for {ticker}")

    # 1. Try yfinance (often faster, but potentially rate-limited for info)
    try:
        stock = yf.Ticker(ticker)
        info = stock.info # Fetching 'info' can be rate-limited
        hist_1d = stock.history(period="2d") # Need 2 days for change_pct

        if not hist_1d.empty and len(hist_1d) >= 2:
            latest_price = hist_1d['Close'].iloc[-1]
            previous_price = hist_1d['Close'].iloc[-2]
            change_pct = ((latest_price - previous_price) / previous_price) * 100

            # Basic volatility from recent history (adjust period as needed)
            hist_1mo = get_historical_data(ticker, period="1mo") # Use cached historical data
            volatility = None
            if not hist_1mo.empty:
                daily_returns = hist_1mo['Close'].pct_change().dropna()
                if not daily_returns.empty:
                     volatility = np.std(daily_returns) * np.sqrt(252) # Annualized

            quote = {
                "ticker": ticker,
                "price": round(latest_price, 2),
                "change_pct": round(change_pct, 2),
                "volatility": round(volatility, 4) if volatility is not None else None,
                "source": "yfinance"
            }
            logging.info(f"Successfully fetched quote for {ticker} from yfinance")
            return quote
        else:
            logging.warning(f"Insufficient yfinance history for quote: {ticker}")

    except Exception as e:
        logging.warning(f"yfinance quote fetch failed for {ticker}: {e}. Trying fallbacks.")
        # Explicitly check for 429 error if yfinance starts throwing them for .info
        if "429" in str(e):
             logging.error(f"yfinance rate limit likely hit for {ticker} info.")

    # 2. Try Finnhub (Fallback 1)
    finnhub_key = api_keys.get("finnhub")
    if finnhub_key:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_key}"
            data = _fetch_with_retry(url) # Use retry wrapper
            if data and data.get("c") is not None and data.get("pc") is not None and data.get("pc") != 0:
                price = data["c"]
                prev_close = data["pc"]
                change_pct = ((price - prev_close) / prev_close) * 100
                # Finnhub quote doesn't directly provide volatility, calculate if needed from historical
                hist_1mo = get_historical_data(ticker, period="1mo") # Use cached historical data
                volatility = None
                if not hist_1mo.empty:
                    daily_returns = hist_1mo['Close'].pct_change().dropna()
                    if not daily_returns.empty:
                        volatility = np.std(daily_returns) * np.sqrt(252) # Annualized

                quote = {
                    "ticker": ticker,
                    "price": round(price, 2),
                    "change_pct": round(change_pct, 2),
                    "volatility": round(volatility, 4) if volatility is not None else None,
                    "source": "finnhub"
                }
                logging.info(f"Successfully fetched quote for {ticker} from Finnhub")
                return quote
            else:
                logging.warning(f"Finnhub data not usable/sufficient for {ticker}")
        except Exception as e:
            logging.error(f"Finnhub fallback error for {ticker}: {e}")

    # 3. Try Alpha Vantage (Fallback 2)
    alphavantage_key = api_keys.get("alphavantage")
    if alphavantage_key:
        try:
            # Using TIME_SERIES_DAILY to get latest close and previous close
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={alphavantage_key}"
            data = _fetch_with_retry(url) # Use retry wrapper
            ts_daily = data.get("Time Series (Daily)")
            if ts_daily and len(ts_daily) >= 2:
                dates = sorted(ts_daily.keys(), reverse=True)
                latest_close = float(ts_daily[dates[0]]['4. close'])
                previous_close = float(ts_daily[dates[1]]['4. close'])
                change_pct = ((latest_close - previous_close) / previous_close) * 100

                # Calculate volatility from daily data (or use cached historical)
                hist_1mo = get_historical_data(ticker, period="1mo") # Use cached historical data
                volatility = None
                if not hist_1mo.empty:
                    daily_returns = hist_1mo['Close'].pct_change().dropna()
                    if not daily_returns.empty:
                        volatility = np.std(daily_returns) * np.sqrt(252) # Annualized
                else: # Fallback if historical cache failed
                     closes = [float(ts_daily[d]['4. close']) for d in dates[:21]] # Approx 1 month
                     daily_returns = pd.Series(closes).pct_change().dropna()
                     if not daily_returns.empty:
                          volatility = np.std(daily_returns) * np.sqrt(252) # Annualized


                quote = {
                    "ticker": ticker,
                    "price": round(latest_close, 2),
                    "change_pct": round(change_pct, 2),
                    "volatility": round(volatility, 4) if volatility is not None else None,
                    "source": "alphavantage"
                }
                logging.info(f"Successfully fetched quote for {ticker} from Alpha Vantage")
                return quote
            else:
                logging.warning(f"Alpha Vantage data insufficient for {ticker}")
        except Exception as e:
            logging.error(f"Alpha Vantage fallback error for {ticker}: {e}")

    # If all sources fail
    logging.error(f"Failed to fetch quote data for {ticker} from all sources.")
    return {
        "ticker": ticker,
        "price": None,
        "change_pct": None,
        "volatility": None,
        "source": "failed"
    }

@st.cache_data(ttl=timedelta(minutes=30)) # Cache info for 30 minutes
def get_stock_info(ticker: str) -> dict:
    """Fetches stock info (like sector) from yfinance with cache."""
    logging.info(f"Fetching stock info for {ticker}")
    try:
        # This call itself can be rate-limited or fail
        stock = yf.Ticker(ticker)
        info = stock.info
        # Add a small delay proactively if making many info calls
        # time.sleep(0.1)
        return info
    except Exception as e:
        logging.error(f"Failed to get yfinance info for {ticker}: {e}")
        # Explicitly check for 429 error if yfinance starts throwing them for .info
        if "429" in str(e):
             logging.error(f"yfinance rate limit likely hit for {ticker} info.")
        return {} # Return empty dict on failure