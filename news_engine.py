"""
24-Hour Overnight Financial News Engine:
Scrapes Google News RSS feeds for target tickers over the past 24 hours,
computes VADER polarity scores (-1.0 to +1.0), and reconciles trading consensus.
"""

import feedparser
from urllib.parse import quote
from typing import Dict, Any, List
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def harvest_overnight_news(ticker_symbol: str) -> Dict[str, Any]:
    clean_ticker = ticker_symbol.replace(".NS", "").replace(".BO", "").strip()
    encoded_query = quote(f"{clean_ticker} stock financial news")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    headlines = []
    compound_scores = []
    sia = SentimentIntensityAnalyzer()

    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:8]:
            title = entry.get("title", "")
            source = entry.get("source", {}).get("title", "Market Feed")
            published = entry.get("published", "")

            scores = sia.polarity_scores(title)
            compound = scores["compound"]
            compound_scores.append(compound)

            if compound >= 0.05:
                sent_tag = "Bullish"
            elif compound <= -0.05:
                sent_tag = "Bearish"
            else:
                sent_tag = "Neutral"

            headlines.append({
                "title": title,
                "source": source,
                "published": published,
                "compound": round(compound, 2),
                "sentiment": sent_tag
            })
    except Exception:
        pass

    avg_score = float(sum(compound_scores) / len(compound_scores)) if compound_scores else 0.0

    if avg_score >= 0.15:
        overall_label = "Bullish Catalyst"
    elif avg_score <= -0.15:
        overall_label = "Bearish Risk"
    else:
        overall_label = "Neutral Phase"

    return {
        "symbol": ticker_symbol,
        "composite_score": round(avg_score, 2),
        "overall_label": overall_label,
        "headlines_count": len(headlines),
        "headlines": headlines,
        "summary": f"Overnight News Sentiment: {overall_label} ({avg_score:+.2f})."
    }

def reconcile_signal_with_news(tech_signal: str, news_score: float) -> str:
    adjusted = tech_signal
    if news_score <= -0.25 and "BUY" in tech_signal:
        adjusted = "NEUTRAL / HOLD (Overnight Bearish News Warning)"
    elif news_score >= +0.25 and "SELL" in tech_signal:
        adjusted = "NEUTRAL / HOLD (Overnight Bullish News Warning)"
    elif news_score >= +0.35 and tech_signal == "BUY":
        adjusted = "STRONG BUY"
    elif news_score <= -0.35 and tech_signal == "SELL":
        adjusted = "STRONG SELL"
    return adjusted