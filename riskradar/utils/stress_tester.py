# riskradar/utils/stress_tester.py
import numpy as np
import pandas as pd
# import yfinance as yf # No longer needed directly
# import requests # No longer needed directly
import streamlit as st
import time
import logging
from .data_fetcher import get_historical_data, get_quote_data # Use centralized fetcher

# --- Configuration ---
CONFIDENCE_LEVEL = 0.95
DAYS = 30 # Simulation horizon
SIMULATIONS = 1000 # Number of Monte Carlo paths

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Removed fallback APIs and throttle_wait - handled by data_fetcher

# ------------------ Utility ------------------
def adjust_volatility(vol, sentiment_score):
    """Adjusts volatility based on sentiment score."""
    if not isinstance(vol, (int, float)) or np.isnan(vol):
         logging.warning("Invalid volatility value passed to adjust_volatility. Returning default.")
         return 0.15 # Return a default reasonable volatility if input is bad

    if sentiment_score < 0.4:
        adj_factor = 1.30 # Increased multiplier for stronger effect
        logging.debug(f"Increasing volatility by {adj_factor-1:.0%} due to low sentiment ({sentiment_score:.3f})")
        return vol * adj_factor
    elif sentiment_score > 0.6:
        adj_factor = 0.80 # Increased multiplier for stronger effect
        logging.debug(f"Decreasing volatility by {1-adj_factor:.0%} due to high sentiment ({sentiment_score:.3f})")
        return vol * adj_factor
    return vol

def simulate_price_paths(ticker, start_price, mu, sigma):
    """Simulates Monte Carlo price paths."""
    if sigma <= 0 or np.isnan(sigma): # Handle zero or invalid sigma
        logging.warning(f"Invalid sigma ({sigma}) for {ticker}. Cannot simulate paths.")
        # Return flat paths or handle error appropriately
        paths = np.full((DAYS, SIMULATIONS), start_price)
        return paths

    dt = 1 / 252 # Time step (daily)
    # S(t+dt) = S(t) * exp( (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z )
    # where Z is standard normal random variable
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Precompute random numbers
    random_shocks = np.random.standard_normal((DAYS - 1, SIMULATIONS))

    # Initialize simulation array
    simulations = np.zeros((DAYS, SIMULATIONS))
    simulations[0] = start_price

    # Run simulation
    for t in range(1, DAYS):
        simulations[t] = simulations[t - 1] * np.exp(drift + diffusion * random_shocks[t - 1])

    return simulations

def calculate_var(price_paths):
    """Calculates Value at Risk (VaR) and returns from simulated paths."""
    if price_paths is None or price_paths.shape[1] == 0 or price_paths[0,0] == 0:
         logging.warning("Invalid price paths for VaR calculation.")
         return np.nan, np.array([])

    start_price = price_paths[0, 0] # Assuming all paths start at the same price
    ending_prices = price_paths[-1]
    returns = (ending_prices - start_price) / start_price
    var = np.percentile(returns, (1 - CONFIDENCE_LEVEL) * 100)
    return var, returns

# ------------------ Stress Test Core ------------------
def run_stress_test(ticker: str, sentiment_score: float, api_keys: dict):
    """
    Runs Monte Carlo stress test for a single ticker.

    Args:
        ticker (str): The stock ticker symbol.
        sentiment_score (float): The combined sentiment score (0 to 1).
        api_keys (dict): Dictionary containing API keys if needed by fetchers.

    Returns:
        dict or None: Simulation results or None if data fetching fails.
    """
    logging.info(f"Running stress test for {ticker} with sentiment {sentiment_score:.3f}")

    # --- Get Data using Centralized Fetcher ---
    # Get historical data (e.g., 6 months for drift and volatility calculation)
    hist = get_historical_data(ticker, period="6mo")

    # Get current price (start price for simulation)
    quote_data = get_quote_data(ticker, api_keys)
    start_price = quote_data.get("price")

    # --- Validate Data ---
    if hist.empty or len(hist) < 30 or start_price is None:
        logging.error(f"Insufficient data for {ticker}. Cannot run stress test.")
        return None # Cannot proceed without sufficient data

    # --- Calculate Inputs for Simulation ---
    try:
        # Use log returns for calculations (more stable for financial modeling)
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()

        if log_returns.empty or len(log_returns) < 2:
             logging.error(f"Not enough return data points for {ticker}.")
             return None

        # Calculate drift (mu) and volatility (sigma) from historical log returns
        # Annualized drift (mu) = mean(log_returns) * 252
        # Annualized volatility (sigma) = std(log_returns) * sqrt(252)
        mu = np.mean(log_returns) * 252
        vol = np.std(log_returns) * np.sqrt(252)

        if np.isnan(mu) or np.isnan(vol):
            logging.error(f"Calculated mu ({mu}) or vol ({vol}) is NaN for {ticker}.")
            return None

    except Exception as e:
        logging.error(f"Error calculating mu/sigma for {ticker}: {e}")
        return None

    # --- Adjust Volatility based on Sentiment ---
    adj_vol = adjust_volatility(vol, sentiment_score)

    # --- Run Simulation ---
    logging.debug(f"Simulating {ticker}: Start Price={start_price:.2f}, Mu={mu:.4f}, Adj Vol={adj_vol:.4f}")
    paths = simulate_price_paths(ticker, start_price, mu, adj_vol)

    # --- Calculate VaR and Return Stats ---
    var, sim_returns = calculate_var(paths)

    # --- Package Results ---
    if np.isnan(var):
         logging.error(f"VaR calculation failed for {ticker}")
         return None

    return {
        "ticker": ticker,
        "start_price": round(start_price, 2),
        "historical_vol": round(vol, 4),
        "sentiment_score": round(sentiment_score, 3),
        "adjusted_vol": round(adj_vol, 4),
        "VaR_95": round(var * 100, 2), # VaR as percentage loss
        "expected_return_mean": round(np.mean(sim_returns) * 100, 2), # Mean simulated return %
        "expected_return_std": round(np.std(sim_returns) * 100, 2), # Std dev of simulated returns %
        "simulated_paths": paths # Keep paths if needed for plotting later
    }