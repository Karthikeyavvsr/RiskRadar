# riskradar/app.py
import os
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import altair as alt
import logging
import numpy as np

# Configure logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Streamlit - %(levelname)s - %(message)s')

# Import updated utils
# (Ensure these imports point to your actual utility files)
from utils.data_fetcher import get_quote_data, get_stock_info, get_historical_data
from utils.market_data import fetch_price_and_volatility # <<<--- THIS LINE WAS MISSING
from models.sentiment_analyzer import SentimentAnalyzer
from utils.sentiment_logger import log_sentiment_score, load_sentiment_log # Keep load_sentiment_log
from utils.stress_tester import run_stress_test
from utils.hedge_advisor import suggest_hedges
from utils.risk_clusterer import get_price_matrix_and_metadata, compute_daily_returns, cluster_assets, plot_clusters
from utils.global_risk_radar import get_global_risk_events # Assuming this uses requests with retry internally if needed
from utils.strategy_simulator import simulate_strategy_and_metrics
from utils.performance_metrics import ( # Import specific metrics if needed directly, otherwise simulator handles it
    compute_cagr, compute_xirr, compute_sharpe_ratio, compute_sortino_ratio,
    compute_max_drawdown, compute_alpha_beta, compute_var
)


# ------------------- DB Setup & Initialization -------------------
DB_FILE = 'data/portfolio.db'
LOG_FILE = "data/sentiment_log.csv" # Define log file path

def init_db():
    # (Keep init_db function as before)
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            weight REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized.")

init_db()

# Initialize session state variables if they don't exist
# This now runs only once when the session starts
if 'initialized' not in st.session_state:
    st.session_state.initialized = True # Mark as initialized
    st.session_state.portfolio_entries = []
    st.session_state.analysis_results = None
    st.session_state.edited_portfolio = pd.DataFrame(columns=['ticker', 'weight']) # Start empty

    # Try loading from DB on first load only
    conn = sqlite3.connect(DB_FILE)
    try:
        df_initial = pd.read_sql_query("SELECT ticker, weight FROM portfolio", conn)
        if not df_initial.empty:
            st.session_state.portfolio_entries = df_initial.to_dict('records')
            st.session_state.edited_portfolio = pd.DataFrame(st.session_state.portfolio_entries)
            logging.info(f"Loaded {len(st.session_state.portfolio_entries)} entries from DB into session state on initial load.")
        else:
             logging.info("No portfolio found in DB on initial load.")
    except Exception as e:
        logging.error(f"Failed to load initial portfolio from DB: {e}")
    finally:
        conn.close()


# ------------------- Load Secrets and Initialize Services -------------------
st.set_page_config(page_title="RiskRadar MVP", layout="wide", initial_sidebar_state="expanded") # Keep sidebar open initially
st.title("📈 RiskRadar MVP")
st.markdown("AI-Powered Portfolio Risk Analysis for Retail Investors")

# (Keep secrets loading as before)
news_api_key = st.secrets.get("news_api_key", "YOUR_NEWS_API_KEY")
reddit_client_id = st.secrets.get("reddit_client_id", "YOUR_REDDIT_CLIENT_ID")
reddit_client_secret = st.secrets.get("reddit_client_secret", "YOUR_REDDIT_SECRET")
reddit_user_agent = st.secrets.get("reddit_user_agent", "RiskRadarApp/0.1 by YourUsername")
finnhub_key = st.secrets.get("finnhub_api_key", "YOUR_FINNHUB_KEY")
alphavantage_key = st.secrets.get("alphavantage_key", "YOUR_ALPHAVANTAGE_KEY")

if any(k.startswith("YOUR_") for k in [news_api_key, reddit_client_id, reddit_client_secret, finnhub_key, alphavantage_key]):
     st.sidebar.warning("API keys missing. Configure secrets.", icon="⚠️")

api_keys = {"finnhub": finnhub_key, "alphavantage": alphavantage_key}

# (Keep analyzer initialization as before)
@st.cache_resource
def get_sentiment_analyzer():
     logging.info("Initializing Sentiment Analyzer...")
     use_finetuned_model = st.secrets.get("use_finetuned_model", False)
     try:
        analyzer = SentimentAnalyzer(
            news_api_key=news_api_key,
            reddit_client_id=reddit_client_id,
            reddit_client_secret=reddit_client_secret,
            reddit_user_agent=reddit_user_agent,
            use_finetuned_model=use_finetuned_model
        )
        logging.info("Sentiment Analyzer initialized.")
        return analyzer
     except Exception as e:
         st.error(f"Failed to initialize Sentiment Analyzer: {e}. Check API keys/credentials.")
         logging.error(f"Sentiment Analyzer init failed: {e}")
         return None

analyzer = get_sentiment_analyzer()

# ------------------- Sidebar: Dynamic Portfolio Builder -------------------
st.sidebar.header("Build Your Portfolio")

popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'META', 'BRK-B']

# --- Input section using st.form ---
st.sidebar.subheader("Add Asset")
with st.sidebar.form("add_asset_form", clear_on_submit=True):
    # (Keep the form content as before)
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker_option = st.selectbox(
            "Select Ticker", options=[""] + popular_tickers, index=0, key="ticker_select_in_form"
        )
        manual_ticker = st.text_input(
             "Or Enter Ticker", key="ticker_manual_in_form").upper().strip()
        selected_ticker = manual_ticker if manual_ticker else ticker_option
    with col2:
        new_weight = st.number_input("Weight %", min_value=0.1, max_value=100.0, value=10.0, step=0.1, key="new_weight_in_form")

    submitted_add = st.form_submit_button("➕ Add Asset")

    if submitted_add:
        # (Keep the add asset logic as before, appending to st.session_state.portfolio_entries
        # and updating st.session_state.edited_portfolio)
        if not selected_ticker:
            st.sidebar.warning("Please select or enter a ticker.")
        elif selected_ticker in [p['ticker'] for p in st.session_state.portfolio_entries]:
            st.sidebar.warning(f"{selected_ticker} is already in the portfolio. Edit weight below.")
        elif new_weight <= 0:
            st.sidebar.warning("Weight must be positive.")
        else:
            st.session_state.portfolio_entries.append({'ticker': selected_ticker, 'weight': new_weight})
            logging.info(f"Added {selected_ticker} ({new_weight}%) to session portfolio via form.")
            st.session_state.edited_portfolio = pd.DataFrame(st.session_state.portfolio_entries)
            # Rerun happens automatically due to form submission state change

# --- Display and Edit Current Portfolio ---
st.sidebar.subheader("Current Portfolio Entries")

if 'edited_portfolio' not in st.session_state or st.session_state.edited_portfolio.empty:
    st.sidebar.info("Add assets using the form above.")
    df_for_editor = pd.DataFrame(columns=['ticker', 'weight'])
else:
    df_for_editor = st.session_state.edited_portfolio

# (Keep the data_editor logic as before, updating st.session_state.portfolio_entries and st.session_state.edited_portfolio)
edited_df = st.sidebar.data_editor(
    df_for_editor,
    key="portfolio_editor",
    num_rows="dynamic",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", required=True),
        "weight": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0, step=0.1, required=True, format="%.1f%%")
    },
    hide_index=True,
    use_container_width=True
)

# Update session state if the editor changed the data
if not edited_df.equals(st.session_state.edited_portfolio):
     edited_df['ticker'] = edited_df['ticker'].str.upper().str.strip()
     edited_df.dropna(subset=['ticker'], inplace=True)
     edited_df['weight'] = pd.to_numeric(edited_df['weight'], errors='coerce').fillna(0.0)
     edited_df = edited_df[edited_df['weight'] > 0]
     edited_df.drop_duplicates(subset=['ticker'], keep='last', inplace=True)
     st.session_state.portfolio_entries = edited_df.to_dict('records')
     st.session_state.edited_portfolio = edited_df
     logging.info("Portfolio updated via data editor.")
     st.rerun()

# Calculate and Display Total Weight
current_display_df = st.session_state.edited_portfolio
total_weight = current_display_df['weight'].sum() if not current_display_df.empty else 0
weight_color = "green" if np.isclose(total_weight, 100.0) else "red"
st.sidebar.markdown(f"**Total Weight:** <span style='color:{weight_color};'>{total_weight:.2f}%</span>", unsafe_allow_html=True)
if not np.isclose(total_weight, 100.0) and not current_display_df.empty:
    st.sidebar.warning("Adjust weights to sum to 100% before analyzing.")


# --- Action Buttons ---
st.sidebar.divider()
col_analyze, col_reset = st.sidebar.columns(2)

# Analyze Button
analysis_disabled = not np.isclose(total_weight, 100.0) or current_display_df.empty
with col_analyze:
    if st.button("🚀 Analyze Portfolio", key="analyze_button", disabled=analysis_disabled, use_container_width=True):
        if analysis_disabled:
            st.error("Cannot analyze. Ensure portfolio is not empty and weights sum to 100%.")
        else:
            final_portfolio_df = st.session_state.edited_portfolio.copy()
            # --- Save Final Portfolio to DB ---
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM portfolio")
                valid_entries = [(row['ticker'], row['weight']) for _, row in final_portfolio_df.iterrows()]
                if valid_entries:
                    cursor.executemany("INSERT INTO portfolio (ticker, weight) VALUES (?, ?)", valid_entries)
                    conn.commit()
                    logging.info(f"Final portfolio saved to DB: {valid_entries}")
                else:
                     logging.warning("Attempted to analyze an empty portfolio.")
            except sqlite3.Error as e:
                conn.rollback()
                st.error(f"Database error saving final portfolio: {e}")
                logging.error(f"DB error saving final portfolio: {e}")
            finally:
                conn.close()

            # --- Trigger Main Analysis ---
            with st.spinner('Performing full portfolio analysis... Please wait.'):
                try:
                    # --- Run All Analysis Steps ---
                    logging.info("Starting full analysis run...")
                    # 1. Market Snapshot - THIS LINE IS NOW CORRECT
                    market_df = fetch_price_and_volatility(final_portfolio_df['ticker'].tolist(), api_keys)

                    # 2. Sentiment Analysis
                    sentiment_data = []
                    if analyzer: # Check if analyzer initialized successfully
                        for ticker in final_portfolio_df['ticker'].tolist():
                            try:
                                result = analyzer.get_sentiment_scores(ticker)
                                sentiment_data.append(result)
                                if result.get('combined_score') is not None:
                                    log_sentiment_score(result['ticker'], result['combined_score'], result['meta'])
                            except Exception as e:
                                logging.error(f"Sentiment analysis failed for {ticker}: {e}")
                                sentiment_data.append({"ticker": ticker, "combined_score": 0.5, "reddit_score": 0.5, "x_score": 0.5, "news_score": 0.5}) # Add defaults
                    else:
                         st.warning("Sentiment Analyzer not available. Skipping sentiment analysis.")
                    sentiment_df = pd.DataFrame(sentiment_data) if sentiment_data else pd.DataFrame(columns=['ticker'])

                    # 3. Stress Test
                    stress_test_results = []
                    if not sentiment_df.empty:
                        for _, row in sentiment_df.iterrows():
                            try:
                                result = run_stress_test(row['ticker'], row['combined_score'], api_keys)
                                if result: stress_test_results.append(result) # Only append if successful
                            except Exception as e:
                                logging.error(f"Stress test failed for {row['ticker']}: {e}")
                    # Create DataFrame from valid results only
                    stress_df = pd.DataFrame([r for r in stress_test_results if r is not None])

                    # 4. Strategy Simulation
                    metrics_df, portfolio_series = simulate_strategy_and_metrics(final_portfolio_df, api_keys, period="1y", benchmark_ticker="SPY")

                    # 5. Hedging
                    hedge_result = suggest_hedges(final_portfolio_df, sentiment_df) if not sentiment_df.empty else {}

                    # 6. Clustering
                    price_matrix, metadata_df = get_price_matrix_and_metadata(final_portfolio_df['ticker'].tolist(), period="3mo")
                    clustered_df = pd.DataFrame()
                    if not price_matrix.empty and not metadata_df.empty:
                        returns_df = compute_daily_returns(price_matrix)
                        valid_tickers_for_returns = returns_df.columns
                        filtered_metadata_df = metadata_df[metadata_df['ticker'].isin(valid_tickers_for_returns)]
                        if not returns_df.empty and not filtered_metadata_df.empty:
                           num_clust = min(max(2, len(final_portfolio_df) // 3), 5)
                           clustered_df = cluster_assets(returns_df, filtered_metadata_df, n_clusters=num_clust)

                    # 7. Global Risk
                    radar_df = get_global_risk_events(alphavantage_key)


                    # Store results in session state
                    st.session_state.analysis_results = {
                         "market_snapshot": market_df, "sentiment": sentiment_df, "stress_test": stress_df,
                         "simulation_metrics": metrics_df, "simulation_series": portfolio_series,
                         "hedging": hedge_result, "clustering": clustered_df, "global_risk": radar_df,
                         "portfolio_analyzed": final_portfolio_df
                    }
                    logging.info("Full analysis complete and results stored in session state.")
                    st.success("Analysis Complete!")

                except Exception as e:
                    st.error(f"An error occurred during the full analysis: {e}")
                    logging.exception("Error during full analysis trigger block:")
                    st.session_state.analysis_results = None

# Reset Button
with col_reset:
    # (Keep reset button logic as before)
    if st.button("🔄 Reset All", key="reset_button", use_container_width=True):
        st.session_state.portfolio_entries = []
        st.session_state.edited_portfolio = pd.DataFrame(columns=['ticker', 'weight'])
        st.session_state.analysis_results = None
        logging.info("Session state portfolio and results cleared.")
        conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM portfolio"); conn.commit()
            logging.info("Portfolio data cleared from database.")
        except sqlite3.Error as e:
            conn.rollback(); st.error(f"Database error during reset: {e}"); logging.error(f"DB error during reset: {e}")
        finally: conn.close()
        try:
            if os.path.exists(LOG_FILE): os.remove(LOG_FILE); logging.info("Sentiment log file cleared.")
        except OSError as e: st.warning(f"Could not clear sentiment log file: {e}"); logging.error(f"Error clearing log file: {e}")
        st.cache_data.clear(); st.cache_resource.clear(); logging.info("Streamlit caches cleared.")
        st.success("Portfolio and analysis reset!")
        st.rerun()


# ------------------- Main Panel: Display Results WITH EXPLANATIONS -------------------
st.header("📊 Portfolio Analysis Results")

# Display Pie Chart Dynamically (based on sidebar edits)
st.subheader("Live Portfolio Allocation Preview")
# (Keep pie chart logic as before)
live_preview_df = st.session_state.edited_portfolio
if not live_preview_df.empty:
     try:
          live_total_weight = live_preview_df['weight'].sum(); plot_weights = live_preview_df['weight']
          if not np.isclose(live_total_weight, 100.0) and live_total_weight > 0: plot_weights = (live_preview_df['weight'] / live_total_weight) * 100
          fig = px.pie(live_preview_df, names='ticker', values=plot_weights, title='Portfolio Allocation Preview', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
          fig.update_traces(textposition='inside', textinfo='percent+label'); st.plotly_chart(fig, use_container_width=True)
     except Exception as e: st.error(f"Could not generate preview pie chart: {e}"); logging.error(f"Preview pie chart error: {e}")
else: st.info("Add assets in the sidebar to see the allocation preview.")
st.divider()

if st.session_state.analysis_results:
    logging.info("Displaying analysis results from session state.")
    results = st.session_state.analysis_results

    # --- Display Analysis Sections WITH EXPLANATIONS ---

    # 1. Market Snapshot
    st.subheader("📉 Market Snapshot")
    st.markdown("Current price, percentage change today, and recent volatility (risk) for each asset.")
    market_df = results.get("market_snapshot", pd.DataFrame())
    if not market_df.empty:
        st.dataframe(market_df.set_index('ticker'), use_container_width=True)
        with st.expander("What does 'Volatility' mean here?"):
            st.markdown("""
            Volatility measures how much an asset's price tends to swing up or down. It's calculated here as the *annualized standard deviation* of daily returns over the past month(s).
            * **Higher Volatility:** Means the price has been making larger moves (more risky/uncertain).
            * **Lower Volatility:** Means the price has been more stable (less risky/uncertain recently).
            This is a backward-looking measure of recent price risk.
            """)
    else: st.warning("Market snapshot data unavailable.")
    st.divider()

    # 2. Sentiment Analysis
    st.subheader("🧠 Social & News Sentiment")
    st.markdown("Measures the overall 'mood' or opinion expressed about each stock on Reddit, X/Twitter, and in recent news articles.")
    sentiment_df = results.get("sentiment", pd.DataFrame())
    if not sentiment_df.empty and 'ticker' in sentiment_df.columns:
        display_sentiment_df = sentiment_df[['ticker', 'reddit_score', 'x_score', 'news_score', 'combined_score']].copy()
        display_sentiment_df.rename(columns={'combined_score': 'Overall Sentiment'}, inplace=True)
        st.dataframe(display_sentiment_df.set_index("ticker"), use_container_width=True)
        with st.expander("How to interpret Sentiment Scores (0 to 1):"):
             st.markdown("""
             The scores estimate the overall sentiment expressed in recent online discussions and news for each ticker. The AI (FinBERT model) reads the text and assigns a score:
             * **Above 0.6 (e.g., 0.6 to 1.0): Generally Bullish / Positive.** Indicates more positive language is being used. High scores might correlate with positive expectations or reactions.
             * **Around 0.4 to 0.6: Generally Neutral.** Indicates a mix of opinions or neutral language.
             * **Below 0.4 (e.g., 0.0 to 0.4): Generally Bearish / Negative.** Indicates more negative language is being used. Low scores might suggest concerns, negative news, or poor expectations.

             **Why care?** Market sentiment, especially from social media, can sometimes influence short-term price movements and volatility, particularly for stocks popular with retail investors. This tool uses the 'Overall Sentiment' score to slightly adjust the risk (volatility) assumed in the Stress Test below.
             """)
        # (Keep sentiment drift plot expander as before)
        with st.expander("📈 View Sentiment Drift Over Time"):
             log_df = load_sentiment_log(); analyzed_portfolio_df = results.get("portfolio_analyzed", pd.DataFrame())
             if not log_df.empty:
                 filtered_log = log_df[log_df['ticker'].isin(analyzed_portfolio_df['ticker'])]
                 if not filtered_log.empty:
                     chart = alt.Chart(filtered_log).mark_line(point=True).encode(x=alt.X('timestamp:T', title='Timestamp'),y=alt.Y('score:Q', title='Sentiment Score', scale=alt.Scale(domain=[0, 1])),color='ticker:N',tooltip=['timestamp:T', 'ticker:N', 'score:Q']).properties(title='Sentiment Score Over Time').interactive()
                     st.altair_chart(chart, use_container_width=True)
                 else: st.write("No historical sentiment data.")
             else: st.write("Sentiment log empty.")
    else: st.warning("Sentiment analysis data unavailable.")
    st.divider()

    # 3. Stress Test
    st.subheader("🚨 Stress Test Simulation (30-Day Monte Carlo)")
    st.markdown("Simulates thousands of possible price movements over the next 30 days to estimate potential risks, considering recent trends and current sentiment.")
    stress_df = results.get("stress_test", pd.DataFrame())
    if not stress_df.empty:
         st.dataframe(stress_df[['ticker', 'start_price', 'historical_vol','sentiment_score', 'adjusted_vol', 'VaR_95', 'expected_return_mean']].set_index('ticker'), use_container_width=True)
         with st.expander("Understanding the Stress Test Results:"):
              st.markdown("""
              This uses a technique called **Monte Carlo simulation** to model uncertainty.
              * **Start Price:** The stock price used as the starting point for the simulations.
              * **Historical Vol:** The stock's measured price risk (volatility) based on the last 6 months.
              * **Sentiment Score:** The score from the analysis above.
              * **Adjusted Vol:** The volatility used in the simulation, slightly increased if sentiment is bad (<0.4) or decreased if sentiment is good (>0.6). This tries to account for current mood affecting near-term risk.
              * **VaR_95 (%) (Value at Risk):** This is a key risk number. It estimates the **maximum percentage loss** you might expect over the next 30 days, 95% of the time, *based on these simulated conditions*. For example, a VaR_95 of **-5.0%** means the simulation suggests there's a 5% chance you could lose *at least* 5% of that stock's value in the next 30 days. **Lower (more negative) VaR indicates higher potential downside risk.**
              * **Expected Return (mean %):** The average percentage return across all 1000 simulations over 30 days. This is just an *average expectation* based on past trends and current volatility.
              * **Return Distribution Plots:** The histograms show the spread of possible 30-day returns from the simulations. A wider spread indicates more uncertainty (higher risk). The VaR_95 value corresponds to the left tail (5th percentile) of this distribution.
              """)
         # (Keep histogram plotting logic as before)
         num_cols = 3; cols = st.columns(num_cols); col_idx = 0
         stress_results_list = stress_df.to_dict('records')
         for result in stress_results_list:
             with cols[col_idx % num_cols]:
                 if 'simulated_paths' in result and result['simulated_paths'] is not None:
                     st.markdown(f"**{result['ticker']} Sim Returns**"); start_p = result['simulated_paths'][0, 0]; end_prices = result['simulated_paths'][-1]; sim_returns = (end_prices - start_p) / start_p
                     fig = px.histogram(sim_returns * 100, nbins=30, title=f"{result['ticker']} (VaR: {result['VaR_95']}%)"); fig.update_layout(showlegend=False, yaxis_title="Frequency", xaxis_title="Simulated Return (%)", height=300, margin=dict(l=10, r=10, t=30, b=10))
                     st.plotly_chart(fig, use_container_width=True)
                 else: st.warning(f"No simulation paths for {result['ticker']} to plot.")
             col_idx += 1
    else: st.warning("Stress test results unavailable.")
    st.divider()

    # 4. Strategy Simulation
    st.subheader("📊 Strategy Backtest Metrics (1 Year)")
    st.markdown("Shows how your current portfolio might have performed over the past year compared to a benchmark (SPY).")
    metrics_df = results.get("simulation_metrics", pd.DataFrame())
    portfolio_series = results.get("simulation_series", pd.Series(dtype=float))
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)
        with st.expander("Understanding Key Backtest Metrics:"):
             st.markdown("""
             These metrics estimate past performance based on historical data. *Past performance does not guarantee future results.*
             * **CAGR (%):** Compound Annual Growth Rate. The average yearly growth rate your portfolio would have achieved over the period. Higher is generally better.
             * **Sharpe Ratio:** Measures return compared to risk (volatility). A higher Sharpe Ratio (e.g., > 1) suggests better risk-adjusted returns compared to a risk-free investment.
             * **Sortino Ratio:** Similar to Sharpe, but only considers downside volatility (risk of losses). Higher is better, especially for risk-averse investors.
             * **Max Drawdown (%):** The largest percentage drop from a peak to a subsequent trough during the backtest period. Lower (less negative) is better, indicating smaller losses during downturns.
             * **VaR 95% (%):** Value at Risk based on historical daily returns over the backtest period. Similar interpretation to the stress test VaR, but looking backward.
             * **Alpha (ann.):** Measures performance relative to the benchmark (SPY) after accounting for market risk (Beta). A positive Alpha suggests the portfolio outperformed what was expected based on its market risk. Higher is better.
             * **Beta:** Measures how much the portfolio tends to move compared to the benchmark (SPY). Beta = 1 means it moves with the market. Beta > 1 means it's more volatile than the market. Beta < 1 means less volatile.
             * **XIRR (%):** Internal Rate of Return, considering cash flows (here simplified to initial investment and final value). Similar to CAGR but accounts for timing if more complex cash flows were involved.
             """)
        if not portfolio_series.empty:
            # (Keep portfolio value plot as before)
            portfolio_plot_df = portfolio_series.reset_index(); portfolio_plot_df.columns = ['Date', 'Portfolio Value (Normalized)']
            value_chart = alt.Chart(portfolio_plot_df).mark_line().encode(x=alt.X('Date', title='Date'),y=alt.Y('Portfolio Value (Normalized)', title='Value (Start = 1)'),tooltip=['Date', 'Portfolio Value (Normalized)']).properties(title='Portfolio Value Over Time (Normalized)').interactive()
            st.altair_chart(value_chart, use_container_width=True)
    else: st.warning("Strategy simulation results unavailable.")
    st.divider()

    # 5. Hedging Suggestions
    st.subheader("🛡️ Hedging Suggestions")
    st.markdown("Suggests ways to potentially reduce risk, especially if negative sentiment is detected.")
    hedge_result = results.get("hedging", {})
    if hedge_result:
        # (Keep hedging display logic as before)
        if hedge_result.get('at_risk'):
             st.markdown("##### 📉 Tickers with Low Sentiment (<0.45)"); risk_df = pd.DataFrame(hedge_result['at_risk']); st.dataframe(risk_df[['ticker', 'sector', 'score', 'suggested_hedges']], use_container_width=True)
        if hedge_result.get('general_hedges'):
             st.markdown("##### 🌐 General Portfolio Hedge Recommendations"); st.info(f"Overall portfolio sentiment might be low or many tickers flagged. Consider these general hedges (like Gold, Bonds, Inverse ETFs): `{', '.join(hedge_result['general_hedges'])}`")
        if not hedge_result.get('at_risk') and not hedge_result.get('general_hedges'): st.success("No immediate hedging concerns based on current sentiment thresholds.")
        with st.expander("What is Hedging?"):
             st.markdown("""
             Hedging means taking an action or position designed to reduce the risk of loss in another position.
             * **Why Hedge?** If analysis suggests high risk (like very negative sentiment or high VaR), adding a hedge might lessen potential losses if the market or specific stocks decline.
             * **How it Works Here:** This tool looks for stocks with low sentiment scores. It then suggests ETFs that generally move *inversely* to that stock's sector (e.g., `SQQQ` for Technology) or broad market hedges (like `SH` which is inverse S&P 500, or `GLD` for Gold) if overall sentiment is poor.
             * **Caution:** Hedging isn't free; inverse ETFs often have fees and 'decay' over time, making them suitable mainly for short-term protection. These are just *suggestions* based on sentiment and sector; thorough research is needed before acting.
             """)
    else: st.warning("Hedging suggestion data unavailable.")
    st.divider()

    # 6. Risk Clustering
    st.subheader("🧲 Asset Risk Clustering (3 Months)")
    st.markdown("Groups your assets based on how similarly their prices have moved recently. Assets in the same cluster tend to react similarly to market events.")
    clustered_df = results.get("clustering", pd.DataFrame())
    if not clustered_df.empty and not clustered_df['cluster'].isnull().all():
        # (Keep plotting logic as before)
        cluster_plot = plot_clusters(clustered_df)
        if cluster_plot: st.pyplot(cluster_plot)
        else: st.warning("Failed to generate cluster plot.")
        with st.expander("How to Use Clusters?"):
             st.markdown("""
             Assets within the same cluster (same color/shape group on the plot) have likely moved together recently.
             * **Diversification Check:** If most of your assets fall into just one cluster, your portfolio might not be well-diversified in terms of how its components react to market news or trends. Ideally, you'd have assets spread across different clusters.
             * **Understanding Relationships:** See which assets behave similarly. This can help understand underlying portfolio risks (e.g., if all your tech stocks are in one cluster, a tech downturn affects them all).
             * **PCA Axes:** The X and Y axes represent abstract combinations of factors driving the correlations (derived from PCA). Assets further apart on the plot are less correlated.
             """)
    else: st.warning("Risk clustering results unavailable or failed.")
    st.divider()

    # 7. Global Risk Radar
    with st.expander("🌍 View Global Risk & Event Radar"):
        st.markdown("Recent news headlines categorized by potential macro risk types (Geopolitical, Economic, Market, etc.). Helps identify broader risks that might affect your portfolio.")
        radar_df = results.get("global_risk", pd.DataFrame())
        if not radar_df.empty:
            # (Keep display logic as before)
            radar_df['time_published'] = pd.to_datetime(radar_df['time_published']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(radar_df[['time_published', 'title', 'categories', 'source']], hide_index=True, use_container_width=True)
        else:
            st.write("Global risk event data unavailable.")

else:
    # This message shows if the analysis hasn't been triggered yet
    st.info("Build or adjust your portfolio in the sidebar. Click 'Analyze Portfolio' when weights sum to 100%.")


# Add footer or information
st.sidebar.markdown("---")
st.sidebar.info("RiskRadar MVP - Dynamic Portfolio Builder")