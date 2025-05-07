import requests
import pandas as pd
import feedparser
from datetime import datetime
import re
import time

# ========== Configuration ==========
ALPHAVANTAGE_NEWS_URL = "https://www.alphavantage.co/query"
RSS_FEEDS = [
    "http://feeds.reuters.com/reuters/businessNews",
    "https://www.ft.com/?format=rss",
    "https://www.bloomberg.com/feed/podcast/etf-report.xml"
]
MAX_ARTICLES = 50

RISK_KEYWORDS = {
    "geopolitical": ["war", "conflict", "nuclear", "tension", "protest", "sanctions"],
    "economic": ["recession", "inflation", "bankruptcy", "slowdown", "GDP", "debt crisis"],
    "natural_disaster": ["earthquake", "hurricane", "wildfire", "flood", "tsunami"],
    "policy": ["rate hike", "interest rate", "central bank", "regulation", "stimulus"],
    "market": ["crash", "volatility", "selloff", "bubble", "correction"]
}

# ========== Functions ==========

def classify_event(text):
    tags = set()
    lower_text = text.lower()
    for category, keywords in RISK_KEYWORDS.items():
        if any(re.search(rf"\b{k}\b", lower_text) for k in keywords):
            tags.add(category)
    return list(tags)

def fetch_alphavantage_news(api_key, tickers=['SPY']):
    all_articles = []
    for symbol in tickers:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": api_key
        }
        try:
            response = requests.get(ALPHAVANTAGE_NEWS_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                feed = data.get("feed", [])
                for item in feed[:MAX_ARTICLES]:
                    all_articles.append({
                        "source": item.get("source"),
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "url": item.get("url"),
                        "time_published": item.get("time_published"),
                        "categories": classify_event(item.get("title", "") + " " + item.get("summary", ""))
                    })
            else:
                print(f"AlphaVantage rate limit or failure ({response.status_code})")
        except Exception as e:
            print(f"Error fetching AlphaVantage news for {symbol}: {e}")
        time.sleep(1.2)  # avoid hitting rate limits aggressively
    return pd.DataFrame(all_articles)

def fetch_rss_fallback():
    all_items = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:MAX_ARTICLES]:
                all_items.append({
                    "source": entry.get("source", {}).get("title", "RSS"),
                    "title": entry.get("title"),
                    "summary": entry.get("summary", entry.get("description", "")),
                    "url": entry.get("link"),
                    "time_published": entry.get("published", datetime.utcnow().isoformat()),
                    "categories": classify_event(entry.get("title", "") + " " + entry.get("summary", ""))
                })
        except Exception as e:
            print(f"Error parsing RSS feed: {url}\n{e}")
    return pd.DataFrame(all_items)

def get_global_risk_events(api_key):
    df = fetch_alphavantage_news(api_key, tickers=['SPY', 'QQQ', 'MSFT'])
    if df.empty or df['categories'].map(len).sum() == 0:
        print("⚠️ Falling back to RSS feeds.")
        df = fetch_rss_fallback()

    df = df[df['categories'].map(len) > 0]  # keep only categorized events
    df["time_published"] = pd.to_datetime(df["time_published"], errors="coerce")
    df = df.dropna(subset=["time_published"])
    return df.sort_values("time_published", ascending=False).reset_index(drop=True)[[
        "time_published", "title", "summary", "categories", "source", "url"
    ]]

# ========== Usage ==========
# from utils.global_risk_radar import get_global_risk_events
# radar_df = get_global_risk_events(st.secrets["alphavantage_key"])
# st.dataframe(radar_df)
