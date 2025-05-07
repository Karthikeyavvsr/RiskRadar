# riskradar/utils/hedge_advisor.py
import yfinance as yf # Keep for Ticker object usage if needed, but info comes from data_fetcher
import pandas as pd
import logging
from collections import Counter
from .data_fetcher import get_stock_info # Import centralized info fetcher

logging.basicConfig(level=logging.INFO)

# Simple mapping of sector to recommended defensive/hedging instruments
SECTOR_HEDGES = {
    "Technology": ["XLK", "VGT", "SQQQ"], # Example: SQQQ is inverse leveraged
    "Consumer Cyclical": ["XLY", "SH"],  # Example: SH is inverse S&P 500
    "Financial Services": ["XLF", "FAZ"], # Example: FAZ is inverse leveraged financials
    "Energy": ["XLE", "DRIP"], # Example: DRIP is inverse leveraged oil/gas
    "Healthcare": ["XLV"],
    "Utilities": ["XLU"],
    "Basic Materials": ["XLB"],
    "Industrials": ["XLI"],
    "Real Estate": ["VNQ", "REK"], # Example: REK is inverse real estate
    "Communication Services": ["XLC"],
    "Consumer Defensive": ["XLP"], # Added Consumer Defensive
    # Consider adding more specific or less common sectors if needed
}

# General market or defensive hedges
DEFAULT_HEDGES = ["GLD", "TLT", "VIXY", "SH", "SPXU"] # SPXU is inverse leveraged S&P 500

def get_sector(ticker):
    """Gets sector using the cached data fetcher."""
    try:
        info = get_stock_info(ticker) # Use cached function
        return info.get("sector", None)
    except Exception as e:
        logging.error(f"Error getting sector for {ticker} via get_stock_info: {e}")
        return None


def suggest_hedges(portfolio_df: pd.DataFrame, sentiment_df: pd.DataFrame):
    """
    Suggests hedges based on sector exposure and sentiment.

    Args:
        portfolio_df: DataFrame with columns 'ticker', 'weight'.
        sentiment_df: DataFrame with columns 'ticker', 'combined_score'.

    Returns:
        dict: Contains 'at_risk' tickers with details and 'general_hedges'.
    """
    hedge_recommendations = []
    sector_map = {}
    risk_flags = []
    low_sentiment_tickers = 0

    logging.info("Generating hedge suggestions...")

    # Ensure sentiment_df has ticker as index for easier lookup
    if 'ticker' in sentiment_df.columns:
        sentiment_lookup = sentiment_df.set_index('ticker')['combined_score']
    else:
        logging.warning("Sentiment DataFrame does not contain 'ticker' column.")
        sentiment_lookup = pd.Series(dtype=float)


    for _, row in portfolio_df.iterrows():
        ticker = row['ticker']
        # Use .get(ticker, 0.5) for safer lookup if ticker might be missing in sentiment_df
        sentiment_score = sentiment_lookup.get(ticker, 0.5)

        sector = get_sector(ticker)
        sector_map[ticker] = sector if sector else "Unknown"

        # Define threshold for low sentiment
        if sentiment_score < 0.45: # Lowered threshold slightly
            low_sentiment_tickers += 1
            hedges = []
            if sector and sector in SECTOR_HEDGES:
                 hedges = SECTOR_HEDGES[sector]
            elif sector: # Sector known but no specific hedge defined
                 hedges = DEFAULT_HEDGES[:2] # Suggest Gold/Bonds
            else: # Unknown sector
                 hedges = DEFAULT_HEDGES # Suggest general hedges

            # Add ticker to risk flags only if specific hedges were found or it's generally low sentiment
            if hedges:
                 risk_flags.append({
                     "ticker": ticker,
                     "sector": sector_map[ticker],
                     "score": round(sentiment_score, 3),
                     "suggested_hedges": hedges # Renamed for clarity
                 })
            logging.info(f"Ticker {ticker} flagged due to low sentiment ({sentiment_score:.3f}). Sector: {sector_map[ticker]}. Suggested Hedges: {hedges}")


    # Recommend general hedges if average sentiment is low OR a significant portion of tickers have low sentiment
    avg_sentiment = sentiment_lookup.mean() if not sentiment_lookup.empty else 0.5
    portfolio_size = len(portfolio_df)
    low_sentiment_ratio = low_sentiment_tickers / portfolio_size if portfolio_size > 0 else 0

    if avg_sentiment < 0.48 or low_sentiment_ratio > 0.4: # Adjusted criteria
        logging.info(f"Overall low sentiment detected (Avg: {avg_sentiment:.3f}, Low Ratio: {low_sentiment_ratio:.2f}). Adding general hedges.")
        # Use set to avoid duplicates if already suggested for specific tickers
        current_general_hedges = set(hedge_recommendations)
        for h in DEFAULT_HEDGES:
             if h not in current_general_hedges:
                  hedge_recommendations.append(h)
    else:
        logging.info(f"Overall sentiment OK (Avg: {avg_sentiment:.3f}, Low Ratio: {low_sentiment_ratio:.2f}).")


    return {
        "at_risk": risk_flags,
        "general_hedges": hedge_recommendations, # Contains only portfolio-wide suggestions now
        "sector_map": sector_map
    }