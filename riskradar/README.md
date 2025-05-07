# RiskRadar: AI-Powered Portfolio Risk Analysis Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) **RiskRadar is an innovative portfolio project demonstrating an AI-powered risk analysis and stress-testing tool designed specifically for retail investors.**

It aims to democratize sophisticated financial modeling by integrating real-time market data, advanced sentiment analysis derived from social media (X/Twitter, Reddit) and news sources, and robust risk metrics into an accessible, interactive web application.


**(Placeholder for Screenshot/GIF)**
## Overview

Traditional retail investing platforms often lack comprehensive risk assessment tools, while institutional platforms (like BlackRock's Aladdin) are inaccessible. RiskRadar bridges this gap by providing:

* **Sentiment-Driven Insights:** Leverages NLP (FinBERT) to analyze market mood from diverse online sources (X, Reddit, News).
* **Advanced Risk Modeling:** Implements Monte Carlo simulations, Value-at-Risk (VaR), and strategy backtesting.
* **AI-Integrated Risk:** Uniquely adjusts simulation parameters (volatility) based on calculated sentiment scores.
* **Actionable Information:** Offers hedging suggestions and risk clustering visualizations.
* **User-Friendly Interface:** Built with Streamlit for interactive analysis and includes explanations for complex metrics.
* **Zero-Cost Philosophy:** Developed using free and open-source technologies and APIs.

## Key Features (Current MVP - v1)

* **Dynamic Portfolio Input:** Interactively add/edit/remove assets (tickers & weights) with real-time validation and allocation preview.
* **Market Snapshot:** Displays current price, daily change, and recent volatility for portfolio assets.
* **Multi-Source Sentiment Analysis:** Aggregates and scores sentiment from X/Twitter, Reddit (r/wallstreetbets), and NewsAPI using FinBERT. Displays overall and source-specific scores, plus historical drift.
* **Sentiment-Adjusted Stress Testing:** Runs Monte Carlo simulations (30-day horizon) with volatility adjusted by sentiment scores, calculating 95% VaR and expected return distributions.
* **Strategy Backtesting:** Simulates portfolio performance over the past year against a benchmark (SPY), providing key metrics (CAGR, Sharpe, Sortino, Max Drawdown, Alpha, Beta).
* **Hedging Suggestions:** Recommends potential hedges (ETFs, defensive assets) based on low sentiment scores and sector analysis.
* **Risk Clustering:** Groups assets based on historical return correlations using PCA and K-Means, visualized to show diversification/concentration.
* **Global Risk Radar:** Monitors news feeds for categorized macro risks (Geopolitical, Economic, Market, etc.).
* **User Explanations:** Integrated expanders provide clear, beginner-friendly explanations for financial metrics (VaR, Sharpe Ratio, etc.).

## Technology Stack

* **Core Language:** Python (3.10+)
* **Web Framework/UI:** Streamlit
* **Data Handling & Numerics:** Pandas, NumPy, SciPy
* **AI/ML:** Hugging Face `transformers`, `torch`, `scikit-learn` (PCA, K-Means)
* **Financial Data APIs:** `yfinance`, `requests` (for Finnhub, Alpha Vantage, NewsAPI), `praw` (Reddit), `feedparser` (RSS)
* **Web Scraping (Sentiment):** `snscrape` (via subprocess - Note: subject to platform changes)
* **API Resilience:** `tenacity`
* **Visualization:** Plotly, Altair, Seaborn, Matplotlib
* **Database:** SQLite3
* **Version Control:** Git / GitHub

## Setup and Installation

Follow these steps to run RiskRadar locally:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Karthikeyavvsr/RiskRadar-MVP.git](https://github.com/Karthikeyavvsr/RiskRadar-MVP.git) # Replace with your repo URL if different
    cd RiskRadar-MVP
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    * **Using Conda:**
        ```bash
        conda create -n riskradar_env python=3.10
        conda activate riskradar_env
        ```
    * **Using venv:**
        ```bash
        python -m venv venv
        # Linux/macOS:
        source venv/bin/activate
        # Windows:
        .\venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r riskradar/requirements.txt
    ```
    *(Note: Installation, especially for PyTorch, might take some time.)*

4.  **Configure API Keys:**
    * Create a directory named `.streamlit` inside the main `RiskRadar-MVP` folder:
        ```bash
        mkdir .streamlit
        ```
    * Inside `.streamlit`, create a file named `secrets.toml`.
    * Add your API keys to `secrets.toml` like this:
        ```toml
        # .streamlit/secrets.toml
        news_api_key = "YOUR_NEWS_API_KEY_HERE"
        reddit_client_id = "YOUR_REDDIT_CLIENT_ID_HERE"
        reddit_client_secret = "YOUR_REDDIT_SECRET_HERE"
        reddit_user_agent = "RiskRadarApp/0.1 by YourUsername" # Customize if needed
        finnhub_api_key = "YOUR_FINNHUB_KEY_HERE"
        alphavantage_key = "YOUR_ALPHAVANTAGE_KEY_HERE"
        ```
    * Replace the placeholders with your actual keys.

5.  **Run the Application:**
    * Ensure your virtual environment is active and you are in the `RiskRadar-MVP` directory.
    * Run the Streamlit app:
        ```bash
        streamlit run riskradar/app.py
        ```

6.  **Access:** Open the local URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

## Usage

1.  **Build Portfolio:** Use the sidebar to add assets. Select or type a ticker, enter the desired weight (%), and click "➕ Add Asset".
2.  **Adjust Portfolio:** Use the "Current Portfolio Entries" table in the sidebar to edit weights directly or remove assets. Ensure the "Total Weight" reaches 100%.
3.  **Analyze:** Once the portfolio weights sum to 100%, click the "🚀 Analyze Portfolio" button.
4.  **Explore Results:** View the analysis results (Market Snapshot, Sentiment, Stress Test, Backtest, Hedging, Clustering, Global Risk) in the main panel. Use the expanders (`▼ What does this mean?`) for detailed explanations of the metrics.
5.  **Reset:** Click the "🔄 Reset All" button in the sidebar to clear the current portfolio, saved data, logs, and caches to start fresh.

## Project Structure
RiskRadar-MVP/
├── data/                 # Holds SQLite DB, logs (created automatically)
├── riskradar/            # Main application source code
│   ├── models/           # AI/ML models (e.g., sentiment_analyzer.py)
│   ├── utils/            # Utility scripts (data fetching, calculations, etc.)
│   ├── app.py            # Main Streamlit application script
│   └── requirements.txt  # Python dependencies
├── .streamlit/
│   └── secrets.toml      # API keys (user must create)
├── .gitignore            # Files/directories ignored by Git
└── README.md             # This file

## Future Enhancements (Planned for v2+)

Based on the [RiskRadar Enhancement Plan](link_to_enhancement_plan_doc_or_wiki), future versions aim to incorporate:

* Advanced AI: Aspect-Based Sentiment Analysis (ABSA), Explainable AI (XAI), Generative AI summaries.
* Enhanced Risk Metrics: Conditional Value-at-Risk (CVaR), Factor Model Exposure.
* Additional Data Sources: SEC EDGAR filings, Google Trends.
* Community Features: User forums/scenario sharing via Firebase.
* Advanced Backtesting: Integration with libraries like `backtesting.py`.
* Potential QuantLib Integration: For institutional-grade modeling (options pricing, advanced simulations).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (or choose another license).



## Author

**Sairam Karthikeya, V V**

* [LinkedIn](https://www.linkedin.com/in/vvsrk1117/)
* [GitHub](https://github.com/Karthikeyavvsr)
