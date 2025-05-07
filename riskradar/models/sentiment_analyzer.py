import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import praw
import requests
import subprocess
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import re
import numpy as np

class SentimentAnalyzer:
    def __init__(self, news_api_key, reddit_client_id, reddit_client_secret, reddit_user_agent, use_finetuned_model=False):
        # Initialize APIs
        self.news_api_key = news_api_key
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )

        # Load FinBERT or custom model
        model_name = "ProsusAI/finbert" if not use_finetuned_model else "your-custom-model-name"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

        # Scoring weights
        self.weights = {"reddit": 0.3, "x": 0.3, "news": 0.4}

        # Hedging/sarcasm patterns
        self.hedge_keywords = ["might", "could", "possibly", "maybe", "likely", "uncertain"]
        self.sarcasm_patterns = [r"!{2,}", r"\?{2,}", r"\".*\"", r"yeah right", r"as if"]

    def get_reddit_posts(self, ticker, limit=50):
        posts = []
        try:
            for submission in self.reddit.subreddit("wallstreetbets").search(ticker, limit=limit):
                posts.append((submission.title or "") + " " + (submission.selftext or ""))
        except Exception as e:
            print(f"[Reddit fetch failed]: {e}")
        return posts

    def get_x_posts(self, ticker, limit=50):
        query = f"{ticker} lang:en"
        cmd = f"snscrape --max-results {limit} twitter-search \"{query}\""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            lines = result.stdout.splitlines()
            return lines[:limit]
        except Exception as e:
            print(f"[X fetch failed]: {e}")
            return []

    def get_news_articles(self, ticker, limit=10):
        url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize={limit}&apiKey={self.news_api_key}"
        try:
            response = requests.get(url)
            articles = response.json().get("articles", [])
            return [f"{a.get('title') or ''} {a.get('description') or ''}" for a in articles]
        except Exception as e:
            print(f"[News fetch failed]: {e}")
            return []

    def analyze_texts(self, texts):
        sentiments = []
        hedge_flags = 0
        sarcasm_flags = 0

        for text in texts:
            try:
                result = self.nlp(text[:512])[0]  # limit to 512 tokens
                score = self.label_to_score(result['label']) * result['score']
                sentiments.append(score)

                if any(h in text.lower() for h in self.hedge_keywords):
                    hedge_flags += 1
                if any(re.search(p, text.lower()) for p in self.sarcasm_patterns):
                    sarcasm_flags += 1
            except Exception as e:
                print(f"[Sentiment parsing failed]: {e}")
                continue

        return {
            "average_score": np.mean(sentiments) if sentiments else 0.5,
            "hedge_flags": hedge_flags,
            "sarcasm_flags": sarcasm_flags
        }

    def label_to_score(self, label):
        if label == "positive": return 1.0
        elif label == "neutral": return 0.5
        elif label == "negative": return 0.0
        return 0.5

    def get_sentiment_scores(self, ticker):
        reddit = self.get_reddit_posts(ticker)
        x = self.get_x_posts(ticker)
        news = self.get_news_articles(ticker)

        reddit_result = self.analyze_texts(reddit)
        x_result = self.analyze_texts(x)
        news_result = self.analyze_texts(news)

        combined = np.average([
            reddit_result['average_score'],
            x_result['average_score'],
            news_result['average_score']
        ], weights=[self.weights['reddit'], self.weights['x'], self.weights['news']])

        return {
            "ticker": ticker,
            "reddit_score": round(reddit_result['average_score'], 3),
            "x_score": round(x_result['average_score'], 3),
            "news_score": round(news_result['average_score'], 3),
            "combined_score": round(combined, 3),
            "meta": {
                "hedge_flags": reddit_result['hedge_flags'] + x_result['hedge_flags'] + news_result['hedge_flags'],
                "sarcasm_flags": reddit_result['sarcasm_flags'] + x_result['sarcasm_flags'] + news_result['sarcasm_flags'],
                "num_posts": {
                    "reddit": len(reddit),
                    "x": len(x),
                    "news": len(news)
                }
            }
        }
