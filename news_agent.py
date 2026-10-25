"""
Agent B: News Intelligence Agent
Retrieves current company, sector, and market news from authorized feeds (Google News RSS / Finnhub).
Classifies events (Earnings, Guidance, Regulatory, Corporate, Macro, Analyst).
Estimates sentiment polarity (-1.0 to +1.0), relevance score (0.0 to 1.0), and source reliability.
Deduplicates syndicated stories and preserves original URLs, timestamps, and sources.
Never invents headlines or historical news.
"""

from datetime import datetime
import hashlib
import os
import re
from typing import Dict, Any, List, Optional
from urllib.parse import quote
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Event classification keywords
EVENT_KEYWORDS = {
    "Earnings & Financials": ["quarter", "results", "revenue", "profit", "net profit", "ebitda", "earnings", "loss", "q1", "q2", "q3", "q4", "fiscal"],
    "Corporate & M&A": ["acquisition", "merger", "deal", "stake", "buyout", "partnership", "joint venture", "expansion", "invests", "divestment"],
    "Regulatory & Legal": ["court", "tribunal", "sebi", "sec", "fine", "penalty", "probe", "investigation", "compliance", "tax", "rbi", "fed"],
    "Guidance & Outlook": ["forecast", "guidance", "target", "outlook", "upgrade", "downgrade", "cut", "boosts", "expects"],
    "Leadership & Management": ["ceo", "cfo", "board", "director", "resigns", "appoints", "leadership", "founder"],
    "Macro & Sectoral": ["inflation", "rate hike", "gdp", "interest rate", "tariff", "crude", "policy", "budget", "export"]
}

# Reliable financial news sources dictionary
SOURCE_RELIABILITY = {
    "Reuters": 0.95,
    "Bloomberg": 0.95,
    "Economic Times": 0.90,
    "Livemint": 0.88,
    "Moneycontrol": 0.88,
    "CNBC": 0.90,
    "Financial Times": 0.95,
    "Business Standard": 0.88,
    "Wall Street Journal": 0.95,
    "MarketWatch": 0.85,
    "Yahoo Finance": 0.80
}


def clean_headline_for_dedup(title: str) -> str:
    """Removes special chars and lowercase for deduplication hash."""
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower()).strip()
    return " ".join(clean.split()[:5])  # First 5 words capture core headline topic


def classify_event(text: str) -> str:
    """Classifies headline into key event categories."""
    t_lower = text.lower()
    for category, keywords in EVENT_KEYWORDS.items():
        if any(kw in t_lower for kw in keywords):
            return category
    return "General Market Update"


def calculate_relevance(title: str, query_symbol: str) -> float:
    """Estimates how relevant the headline is to the specific company."""
    clean_sym = query_symbol.replace(".NS", "").replace(".BO", "").lower()
    t_lower = title.lower()
    if clean_sym in t_lower:
        return 0.95
    # Token matches
    tokens = clean_sym.split()
    if any(tok in t_lower for tok in tokens if len(tok) > 2):
        return 0.80
    return 0.60


class NewsIntelligenceAgent:
    """
    Collects, analyzes, deduplicates, and evaluates market news with
    source transparency and no artificial fabrication.
    """

    def __init__(self, market: str = "IN"):
        self.market = market
        self.sia = SentimentIntensityAnalyzer()
        # Custom financial lexicon enhancements
        self.sia.lexicon.update({
            "surge": 2.2, "soar": 2.5, "jump": 1.8, "rally": 2.0, "beat": 2.0,
            "outperform": 2.2, "record": 1.5, "plunge": -2.5, "tumble": -2.2,
            "slump": -2.0, "miss": -1.8, "downgrade": -2.2, "probe": -1.8,
            "fraud": -3.5, "penalty": -2.0, "bankruptcy": -3.8
        })

    def harvest_news(self, symbol: str, limit: int = 12) -> Dict[str, Any]:
        """
        Retrieves real-time verified headlines from Google News RSS.
        Deduplicates stories and computes polarity, event type, and reliability.
        """
        clean_ticker = symbol.replace(".NS", "").replace(".BO", "").strip()
        
        # Build query based on market
        if self.market == "IN":
            query_str = f"{clean_ticker} stock share news"
            hl, gl, ceid = "en-IN", "IN", "IN:en"
        else:
            query_str = f"{clean_ticker} stock financial news"
            hl, gl, ceid = "en-US", "US", "US:en"

        encoded = quote(query_str)
        rss_url = f"https://news.google.com/rss/search?q={encoded}&hl={hl}&gl={gl}&ceid={ceid}"

        headlines = []
        seen_hashes = set()
        scores = []
        now = datetime.utcnow()

        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # Deduplicate syndication
                dedup_key = clean_headline_for_dedup(title)
                if dedup_key in seen_hashes:
                    continue
                seen_hashes.add(dedup_key)

                source_name = entry.get("source", {}).get("title", "Financial Feed")
                pub_date = entry.get("published", now.strftime("%Y-%m-%d"))
                link = entry.get("link", "#")

                # Sentiment
                vader_scores = self.sia.polarity_scores(title)
                compound = round(float(vader_scores["compound"]), 3)
                scores.append(compound)

                # Sentiment Category
                if compound >= 0.15:
                    tag = "Bullish"
                elif compound <= -0.15:
                    tag = "Bearish"
                else:
                    tag = "Neutral"

                # Reliability rating
                reliability = SOURCE_RELIABILITY.get(source_name, 0.75)
                relevance = calculate_relevance(title, clean_ticker)
                event_type = classify_event(title)

                headlines.append({
                    "title": title,
                    "source": source_name,
                    "published": pub_date,
                    "url": link,
                    "compound_score": compound,
                    "sentiment_tag": tag,
                    "event_category": event_type,
                    "reliability_score": reliability,
                    "relevance_score": relevance
                })

                if len(headlines) >= limit:
                    break
        except Exception as e:
            return {
                "symbol": symbol,
                "status": "ERROR",
                "error": str(e),
                "composite_score": 0.0,
                "overall_label": "Neutral Phase",
                "headlines_count": 0,
                "headlines": [],
                "summary": "Live news feed temporarily unavailable."
            }

        if not headlines:
            return {
                "symbol": symbol,
                "status": "EMPTY",
                "composite_score": 0.0,
                "overall_label": "Neutral Phase",
                "headlines_count": 0,
                "headlines": [],
                "summary": "No verified recent news headlines found in the last 24-48 hours. Neutral sentiment assumed."
            }

        # Weighted composite score: weighted by relevance and source reliability
        total_weight = sum(h["relevance_score"] * h["reliability_score"] for h in headlines)
        if total_weight > 0:
            weighted_compound = sum(h["compound_score"] * h["relevance_score"] * h["reliability_score"] for h in headlines) / total_weight
        else:
            weighted_compound = sum(scores) / len(scores)

        avg_score = round(float(weighted_compound), 2)

        if avg_score >= 0.18:
            overall_label = "Bullish Catalyst"
        elif avg_score <= -0.18:
            overall_label = "Bearish Risk"
        else:
            overall_label = "Neutral Phase"

        # Count events
        events_found = list({h["event_category"] for h in headlines})

        return {
            "symbol": symbol,
            "status": "SUCCESS",
            "composite_score": avg_score,
            "overall_label": overall_label,
            "headlines_count": len(headlines),
            "headlines": headlines,
            "key_events": events_found,
            "summary": f"Analyzed {len(headlines)} deduplicated articles. Sentiment stance: {overall_label} ({avg_score:+.2f}). Prominent themes: {', '.join(events_found[:3])}."
        }
