# riskradar/utils/strategy_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Import centralized data fetcher and performance metrics
from .data_fetcher import get_historical_data
from .performance_metrics import (
    compute_cagr,
    compute_xirr, # Assuming compute_xirr handles datetime objects correctly
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_alpha_beta,
    compute_var
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Removed old data fetching fallbacks - handled by data_fetcher

# ---------------------- Simulation Core ----------------------

def simulate_strategy(tickers: list, weights: list, api_keys: dict, period: str = "1y", benchmark_ticker: str = "SPY"):
    """
    Simulates portfolio performance based on historical data and calculates metrics.

    Args:
        tickers (list): List of ticker symbols.
        weights (list): List of corresponding portfolio weights (summing to 100).
        api_keys (dict): API keys (passed to data_fetcher if needed, though maybe not for historical).
        period (str): Historical period for simulation (e.g., "1y", "3y").
        benchmark_ticker (str): Ticker for benchmark comparison.

    Returns:
        tuple: (dict of performance metrics, pd.Series of portfolio value over time)
               Returns ({}, pd.Series(dtype=float)) on failure.
    """
    logging.info(f"Simulating strategy for tickers: {tickers} over period: {period}")
    aligned_prices = {}
    valid_tickers = []
    valid_weights = []

    # Fetch historical data for each ticker using the centralized fetcher
    for i, ticker in enumerate(tickers):
        hist_data = get_historical_data(ticker, period=period)
        if not hist_data.empty and 'Close' in hist_data.columns:
            aligned_prices[ticker] = hist_data['Close']
            valid_tickers.append(ticker)
            # Ensure weight corresponds to the successfully fetched ticker
            if i < len(weights):
                 valid_weights.append(weights[i])
            else:
                 logging.warning(f"Weight missing for ticker {ticker}. Assuming 0.")
                 valid_weights.append(0)
        else:
            logging.warning(f"No historical data for {ticker} in period {period}, skipping.")

    # Check if we have any valid data
    if not aligned_prices:
        logging.error("No valid historical data fetched for any ticker. Cannot simulate.")
        return {}, pd.Series(dtype=float)

    # Adjust weights if some tickers were skipped
    total_valid_weight = sum(valid_weights)
    if total_valid_weight == 0:
         logging.error("Sum of weights for valid tickers is zero. Cannot simulate.")
         return {}, pd.Series(dtype=float)
    if total_valid_weight != 100:
         logging.warning(f"Original weights sum to {sum(weights)}. Adjusting weights for valid tickers (sum={total_valid_weight}) to sum to 100.")
         adjusted_weights = [(w / total_valid_weight) * 100 for w in valid_weights]
         logging.info(f"Adjusted weights: {dict(zip(valid_tickers, adjusted_weights))}")
    else:
         adjusted_weights = valid_weights # Weights were already correct or summed to 100

    # Combine into a DataFrame and align timestamps
    price_df = pd.concat(aligned_prices, axis=1).sort_index()
    # Forward fill and then backfill to handle missing values robustly
    price_df = price_df.ffill().bfill()
    price_df = price_df.dropna() # Drop any remaining NaNs (shouldn't happen with ffill/bfill)

    if price_df.empty or len(price_df) < 2:
        logging.error("Price DataFrame is empty or has insufficient data after alignment.")
        return {}, pd.Series(dtype=float)

    # --- Portfolio Calculation ---
    logging.debug("Calculating portfolio value series...")
    # Normalize prices to start at 1 (or initial investment)
    norm_prices = price_df / price_df.iloc[0]
    # Ensure weights array matches columns after potential ticker skips
    weight_arr = np.array(adjusted_weights) / 100.0
    # Calculate weighted portfolio value over time
    portfolio_value_series = (norm_prices * weight_arr).sum(axis=1)
    # Calculate daily returns
    portfolio_returns = portfolio_value_series.pct_change().dropna()

    if portfolio_returns.empty:
        logging.error("Portfolio returns series is empty.")
        return {}, pd.Series(dtype=float)


    # --- Benchmark Comparison ---
    logging.debug(f"Fetching benchmark data: {benchmark_ticker}")
    benchmark_series = get_historical_data(benchmark_ticker, period=period)['Close']
    benchmark_returns = pd.Series(dtype=float) # Initialize
    if not benchmark_series.empty:
        # Align benchmark data to portfolio dates
        benchmark_series = benchmark_series.reindex(portfolio_value_series.index).ffill().bfill()
        if not benchmark_series.isnull().all():
             benchmark_returns = benchmark_series.pct_change().dropna()

    # --- Calculate Metrics ---
    logging.debug("Calculating performance metrics...")
    # Prepare cash flows for XIRR (example: $100 initial investment)
    initial_investment = 100.0
    # Use actual datetime objects from the index
    cash_flow_dates = [portfolio_value_series.index[0].to_pydatetime(), portfolio_value_series.index[-1].to_pydatetime()]
    cash_flows_values = [-initial_investment, portfolio_value_series.iloc[-1] * initial_investment] # End value based on normalized series * initial

    metrics = {}
    try:
        metrics["CAGR (%)"] = round(compute_cagr(portfolio_returns) * 100, 2)
        # Ensure compute_xirr can handle the datetime objects and values
        xirr_val = compute_xirr(list(zip(cash_flow_dates, cash_flows_values)))
        metrics["XIRR (%)"] = round(xirr_val * 100, 2) if not np.isnan(xirr_val) else None
        metrics["Sharpe Ratio"] = round(compute_sharpe_ratio(portfolio_returns), 3)
        metrics["Sortino Ratio"] = round(compute_sortino_ratio(portfolio_returns), 3)
        metrics["Max Drawdown (%)"] = round(compute_max_drawdown(portfolio_value_series) * 100, 2) # Use value series for drawdown
        metrics["VaR 95% (%)"] = round(compute_var(portfolio_returns) * 100, 2) # VaR on returns

        if not benchmark_returns.empty:
            alpha, beta = compute_alpha_beta(portfolio_returns, benchmark_returns)
            metrics["Alpha (ann.)"] = round(alpha * 252, 4) if not np.isnan(alpha) else None # Annualize alpha
            metrics["Beta"] = round(beta, 3) if not np.isnan(beta) else None
        else:
            logging.warning(f"Benchmark data ({benchmark_ticker}) unavailable or empty. Skipping Alpha/Beta.")
            metrics.update({"Alpha (ann.)": None, "Beta": None})

    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        # Return partial metrics if possible, or empty dict on major failure
        return metrics, portfolio_value_series # Return what we have


    logging.info("Strategy simulation and metric calculation complete.")
    return metrics, portfolio_value_series # Return calculated metrics and the value series

# ---------------------- Streamlit-Friendly Wrapper ----------------------

# This function remains largely the same, just calls the updated simulate_strategy
def simulate_strategy_and_metrics(portfolio_df, api_keys, period="1y", benchmark_ticker="SPY"):
    """
    Wrapper function for Streamlit app to simulate strategy and format metrics.
    """
    if portfolio_df.empty:
        logging.warning("Portfolio DataFrame is empty in wrapper function.")
        return pd.DataFrame(), pd.Series(dtype=float)

    tickers = portfolio_df["ticker"].tolist()
    weights = portfolio_df["weight"].tolist() # Ensure weights are passed correctly

    metrics, portfolio_series = simulate_strategy(tickers, weights, api_keys, period=period, benchmark_ticker=benchmark_ticker)

    if not metrics:
        logging.error("Strategy simulation failed in wrapper function.")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Convert metrics dict to DataFrame for display
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    return metrics_df.set_index("Metric"), portfolio_series