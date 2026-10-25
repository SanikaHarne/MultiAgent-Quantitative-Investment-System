"""
Agent A: Market Data Agent
Retrieves authorized real quotes and historical OHLCV data for Indian (NSE/BSE)
and US (NYSE/NASDAQ) equities and indices.
Normalizes symbols, timestamps, currencies, and candle formats.
Detects stale, missing, inconsistent, or insufficient data.
"""

from datetime import datetime, timedelta
import functools
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# Broad searchable instrument catalogue
CATALOGUE_IN = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy & Conglomerate", "exchange": "NSE", "currency": "INR"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "Information Technology", "exchange": "NSE", "currency": "INR"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Financial Services", "exchange": "NSE", "currency": "INR"},
    {"symbol": "INFY.NS", "name": "Infosys Ltd", "sector": "Information Technology", "exchange": "NSE", "currency": "INR"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Financial Services", "exchange": "NSE", "currency": "INR"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecommunications", "exchange": "NSE", "currency": "INR"},
    {"symbol": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG & Conglomerate", "exchange": "NSE", "currency": "INR"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Infrastructure", "exchange": "NSE", "currency": "INR"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Financial Services", "exchange": "NSE", "currency": "INR"},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "sector": "Metals & Mining", "exchange": "NSE", "currency": "INR"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "Consumer Defensives", "exchange": "NSE", "currency": "INR"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Financial Services", "exchange": "NSE", "currency": "INR"},
    {"symbol": "WIPRO.NS", "name": "Wipro Ltd", "sector": "Information Technology", "exchange": "NSE", "currency": "INR"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki", "sector": "Automotive", "exchange": "NSE", "currency": "INR"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Healthcare", "exchange": "NSE", "currency": "INR"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank Ltd", "sector": "Financial Services", "exchange": "NSE", "currency": "INR"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automotive", "exchange": "NSE", "currency": "INR"},
    {"symbol": "NTPC.NS", "name": "NTPC Ltd", "sector": "Utilities & Power", "exchange": "NSE", "currency": "INR"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Financial Services", "exchange": "NSE", "currency": "INR"},
    {"symbol": "TITAN.NS", "name": "Titan Company", "sector": "Consumer Goods", "exchange": "NSE", "currency": "INR"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Consumer Goods", "exchange": "NSE", "currency": "INR"},
    {"symbol": "NIFTYBEES.NS", "name": "Nippon ETF Nifty BeES", "sector": "Index ETF", "exchange": "NSE", "currency": "INR"},
    {"symbol": "GOLDBEES.NS", "name": "Nippon ETF Gold BeES", "sector": "Commodity ETF", "exchange": "NSE", "currency": "INR"}
]

CATALOGUE_US = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology & Cloud", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Communication Services", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Semiconductors", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "META", "name": "Meta Platforms", "sector": "Communication Services", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive / Clean Energy", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway", "sector": "Financial & Conglomerate", "exchange": "NYSE", "currency": "USD"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services", "exchange": "NYSE", "currency": "USD"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Fintech / Payments", "exchange": "NYSE", "currency": "USD"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "exchange": "NYSE", "currency": "USD"},
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive", "exchange": "NYSE", "currency": "USD"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Semiconductors", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Entertainment", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "DIS", "name": "Walt Disney Co.", "sector": "Entertainment", "exchange": "NYSE", "currency": "USD"},
    {"symbol": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "Index ETF", "exchange": "NYSE", "currency": "USD"},
    {"symbol": "QQQ", "name": "Invesco QQQ ETF", "sector": "Tech Index ETF", "exchange": "NASDAQ", "currency": "USD"}
]

# Simple in-memory cache for market quotes and candles with TTL
_CACHE_OHLCV: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
_CACHE_QUOTES: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 180  # 3 minutes for quotes/candles


def normalize_symbol(symbol: str, market: str = "IN") -> str:
    """Normalizes symbol based on market. Appends .NS if Indian ticker lacks exchange."""
    s = symbol.strip().upper()
    if market == "IN":
        if not s.endswith(".NS") and not s.endswith(".BO") and not s.startswith("^"):
            return f"{s}.NS"
    return s


def get_catalog(market: str = "IN") -> List[Dict[str, Any]]:
    """Returns the catalog of vetted instruments."""
    return CATALOGUE_IN if market == "IN" else CATALOGUE_US


def search_catalog(query: str, market: str = "IN") -> List[Dict[str, Any]]:
    """Searches catalog by ticker or company name. If not found, builds candidate object."""
    q = query.strip().upper()
    cat = get_catalog(market)
    matches = [c for c in cat if q in c["symbol"] or q.lower() in c["name"].lower()]
    if matches:
        return matches

    # Build dynamic entry for searched ticker
    norm = normalize_symbol(q, market)
    curr = "INR" if market == "IN" else "USD"
    exch = "NSE" if market == "IN" else "US"
    return [{"symbol": norm, "name": q, "sector": "Equities", "exchange": exch, "currency": curr}]


class MarketDataAgent:
    """
    Market Data Agent encapsulates fetching, normalizing, and validating
    real-time quotes, OHLCV candles, and benchmark indices.
    """

    def __init__(self, market: str = "IN"):
        self.market = market
        self.currency = "INR" if market == "IN" else "USD"
        self.currency_sym = "₹" if market == "IN" else "$"
        self.benchmark_ticker = "^NSEI" if market == "IN" else "^GSPC"
        self.benchmark_name = "Nifty 50" if market == "IN" else "S&P 500"

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Fetches historical OHLCV data. Validates candle integrity and adds return series.
        """
        clean_sym = normalize_symbol(symbol, self.market)
        cache_key = f"{clean_sym}_{period}_{interval}"

        now = datetime.utcnow()
        if not force_refresh and cache_key in _CACHE_OHLCV:
            cached_time, cached_df = _CACHE_OHLCV[cache_key]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return {
                    "symbol": clean_sym,
                    "df": cached_df.copy(),
                    "source": "Cache (Recent Session)",
                    "timestamp": cached_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "status": "SUCCESS",
                    "rows_count": len(cached_df),
                    "is_stale": False,
                    "warnings": []
                }

        warnings = []
        try:
            ticker_obj = yf.Ticker(clean_sym)
            df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
            
            if df.empty or len(df) < 30:
                return {
                    "symbol": clean_sym,
                    "df": pd.DataFrame(),
                    "source": "yfinance",
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "status": "INSUFFICIENT_DATA",
                    "rows_count": len(df),
                    "is_stale": True,
                    "warnings": [f"Insufficient historical bars returned ({len(df)} bars). Expected minimum 30."]
                }

            # Ensure datetime index is clean and localized/timezone-stripped for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Compute daily log and percentage returns
            df["Returns"] = df["Close"].pct_change()
            df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

            # Check staleness
            latest_bar_date = df.index[-1]
            days_old = (now.date() - latest_bar_date.date()).days
            if days_old > 4:
                warnings.append(f"Latest candle is {days_old} days old (Market closed or weekend/holiday).")

            _CACHE_OHLCV[cache_key] = (now, df)

            return {
                "symbol": clean_sym,
                "df": df,
                "source": "yfinance (Real-Time / Recent Market Close)",
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "status": "SUCCESS",
                "rows_count": len(df),
                "is_stale": days_old > 4,
                "warnings": warnings
            }
        except Exception as e:
            return {
                "symbol": clean_sym,
                "df": pd.DataFrame(),
                "source": "None",
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "status": "ERROR",
                "rows_count": 0,
                "is_stale": True,
                "warnings": [f"Failed to fetch market data: {str(e)}"]
            }

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches current or latest verified market quote with data freshness indicators.
        """
        clean_sym = normalize_symbol(symbol, self.market)
        now = datetime.utcnow()

        if clean_sym in _CACHE_QUOTES:
            cached_time, cached_quote = _CACHE_QUOTES[clean_sym]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS and cached_quote.get("last_price", 0) > 0:
                return cached_quote

        ohlcv_res = self.fetch_ohlcv(clean_sym, period="5d", interval="1d")
        df = ohlcv_res.get("df", pd.DataFrame())

        # FALLBACK: If yfinance failed or returned empty data
        if df.empty or len(df) < 2 or "Close" not in df.columns or df["Close"].dropna().empty:
            cat_matches = [c for c in CATALOGUE_IN + CATALOGUE_US if c["symbol"] == clean_sym]
            fallback_price = 2980.0 if "RELIANCE" in clean_sym else 100.0
            if cat_matches:
                # Use approximate price from config catalog if available
                from config import ASSETS_IN, ASSETS_US
                all_assets = ASSETS_IN["stocks"] + ASSETS_US["stocks"]
                found = [a for a in all_assets if a["symbol"] == clean_sym]
                if found:
                    fallback_price = found[0].get("approx_price", 100.0)

            return {
                "symbol": clean_sym,
                "name": clean_sym.replace(".NS", "").replace(".BO", ""),
                "last_price": fallback_price,
                "day_change": 0.0,
                "day_change_pct": 0.0,
                "high_52w": fallback_price * 1.15,
                "low_52w": fallback_price * 0.85,
                "volume": 0,
                "currency": self.currency,
                "currency_sym": self.currency_sym,
                "status": "FALLBACK",
                "source": "Catalog Baseline (Market Feed Unavailable)",
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "is_live": False
            }

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        last_close = float(last_row["Close"])
        prev_close = float(prev_row["Close"])
        change = round(last_close - prev_close, 2)
        change_pct = round(((last_close - prev_close) / prev_close) * 100.0, 2)

        long_res = self.fetch_ohlcv(clean_sym, period="1y", interval="1d")
        df_1y = long_res.get("df", df)
        high_52w = float(df_1y["High"].max()) if not df_1y.empty else last_close * 1.15
        low_52w = float(df_1y["Low"].min()) if not df_1y.empty else last_close * 0.85

        quote = {
            "symbol": clean_sym,
            "name": clean_sym.replace(".NS", "").replace(".BO", ""),
            "last_price": round(last_close, 2),
            "day_change": change,
            "day_change_pct": change_pct,
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "volume": int(last_row.get("Volume", 0)),
            "currency": self.currency,
            "currency_sym": self.currency_sym,
            "status": "ACTIVE",
            "source": "yfinance Market Feed",
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "is_live": not ohlcv_res.get("is_stale", False)
        }

        _CACHE_QUOTES[clean_sym] = (now, quote)
        return quote

    def get_benchmark_series(self, period: str = "2y") -> pd.DataFrame:
        """Fetches the benchmark index daily series (Nifty 50 or S&P 500)."""
        res = self.fetch_ohlcv(self.benchmark_ticker, period=period, interval="1d")
        return res.get("df", pd.DataFrame())
