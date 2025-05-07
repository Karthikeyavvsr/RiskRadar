import os
import pandas as pd
from datetime import datetime

LOG_FILE = "data/sentiment_log.csv"

def log_sentiment_score(ticker, score, meta):
    os.makedirs("data", exist_ok=True)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": now,
        "ticker": ticker,
        "score": score,
        "hedge_flags": meta.get("hedge_flags", 0),
        "sarcasm_flags": meta.get("sarcasm_flags", 0),
    }
    df = pd.DataFrame([entry])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def load_sentiment_log():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE, parse_dates=['timestamp'])
    return pd.DataFrame(columns=['timestamp', 'ticker', 'score', 'hedge_flags', 'sarcasm_flags'])