"""
Kaggle Dataset Loader & Universal Parser Module:
Supports pre-loaded Kaggle financial datasets (Nifty 50, S&P 500, Mutual Funds)
and parses user-uploaded Kaggle CSV files (both wide time-series and long transactional formats).
Computes empirical CAGR, volatility, and covariance matrices directly from Kaggle data.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PRESET_DATASETS = {
    "kaggle_nifty50": {
        "name": "Kaggle Nifty-50 Historical Equities Dataset (5 Years)",
        "file": os.path.join(DATA_DIR, "kaggle_nifty50_historical.csv"),
        "market": "IN",
        "currency": "INR",
        "description": "Daily adjusted closing prices for top NSE Nifty 50 companies (1,240+ trading days).",
        "source_tag": "Kaggle: NIFTY-50 Time-Series"
    },
    "kaggle_sp500": {
        "name": "Kaggle S&P 500 Historical Equities Dataset (5 Years)",
        "file": os.path.join(DATA_DIR, "kaggle_sp500_historical.csv"),
        "market": "US",
        "currency": "USD",
        "description": "Daily adjusted closing prices for S&P 500 mega-caps (1,250+ trading days).",
        "source_tag": "Kaggle: S&P 500 Historical Data"
    },
    "kaggle_mutual_funds": {
        "name": "Kaggle Indian Mutual Funds Performance Dataset",
        "file": os.path.join(DATA_DIR, "kaggle_mutual_funds_india.csv"),
        "market": "IN",
        "currency": "INR",
        "description": "AMFI mutual fund scheme ratings, historical 3Y/5Y CAGR, AUM, and volatility metrics.",
        "source_tag": "Kaggle: Indian Mutual Funds"
    }
}


def load_preset_kaggle_dataset(preset_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads one of the pre-packaged Kaggle benchmark datasets.
    """
    preset = PRESET_DATASETS.get(preset_key, PRESET_DATASETS["kaggle_nifty50"])
    file_path = preset["file"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Kaggle dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date").sort_index()

    metadata = {
        "key": preset_key,
        "name": preset["name"],
        "market": preset["market"],
        "currency": preset["currency"],
        "description": preset["description"],
        "source_tag": preset["source_tag"],
        "rows": len(df),
        "columns": list(df.columns),
        "start_date": str(df.index.min().strftime("%Y-%m-%d")) if isinstance(df.index, pd.DatetimeIndex) and not df.empty else "N/A",
        "end_date": str(df.index.max().strftime("%Y-%m-%d")) if isinstance(df.index, pd.DatetimeIndex) and not df.empty else "N/A"
    }
    return df, metadata


def parse_custom_kaggle_csv(file_buffer) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses any user-uploaded CSV file from Kaggle.
    Automatically detects:
      - Wide format: Date column + multiple asset ticker price columns
      - Long format: Date column, Ticker/Symbol column, and Close/Adj Close price column
    """
    raw_df = pd.read_csv(file_buffer)
    col_map = {str(c).lower().strip(): c for c in raw_df.columns}

    # Identify date column
    date_col = None
    for cand in ["date", "datetime", "timestamp", "time"]:
        if cand in col_map:
            date_col = col_map[cand]
            break

    # Identify if long format (Symbol + Close)
    symbol_col = None
    for cand in ["symbol", "ticker", "stock", "company", "asset"]:
        if cand in col_map:
            symbol_col = col_map[cand]
            break

    price_col = None
    for cand in ["close", "adj close", "adj_close", "price", "nav", "last"]:
        if cand in col_map:
            price_col = col_map[cand]
            break

    # Case A: Long format (e.g. Date, Symbol, Close)
    if date_col and symbol_col and price_col:
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
        raw_df = raw_df.dropna(subset=[date_col, symbol_col, price_col])
        pivoted_df = raw_df.pivot_table(index=date_col, columns=symbol_col, values=price_col, aggfunc="last")
        price_df = pivoted_df.sort_index().ffill().dropna(how="all")
    # Case B: Wide format (Date + Asset columns)
    elif date_col:
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
        raw_df = raw_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        # Keep numeric price columns
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        price_df = raw_df[numeric_cols].ffill().dropna(how="all")
    else:
        # Fallback: assume first column is index or date
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        price_df = raw_df[numeric_cols].ffill().dropna(how="all")

    metadata = {
        "key": "custom_uploaded",
        "name": getattr(file_buffer, "name", "User Uploaded Kaggle Dataset"),
        "rows": len(price_df),
        "columns": list(price_df.columns),
        "start_date": str(price_df.index.min().strftime("%Y-%m-%d")) if isinstance(price_df.index, pd.DatetimeIndex) and not price_df.empty else "N/A",
        "end_date": str(price_df.index.max().strftime("%Y-%m-%d")) if isinstance(price_df.index, pd.DatetimeIndex) and not price_df.empty else "N/A",
        "source_tag": f"Custom Kaggle CSV ({getattr(file_buffer, 'name', 'File')})"
    }
    return price_df, metadata


def build_universe_from_kaggle_data(
    price_df: pd.DataFrame,
    metadata: Dict[str, Any],
    fallback_universe: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transforms a Kaggle price DataFrame into the full structured asset universe
    with empirical CAGR, volatility, and covariance matrix.
    """
    symbols = list(price_df.columns)[:12]  # Focus on top 12 assets
    daily_returns_df = price_df[symbols].pct_change().dropna()
    empirical_cov = daily_returns_df.cov() * 252.0 if len(daily_returns_df) > 20 else None

    enriched_stocks = []
    symbol_meta_lookup = {s["symbol"]: s for s in fallback_universe.get("stocks", [])}

    for sym in symbols:
        series = price_df[sym].dropna()
        if len(series) >= 2:
            latest = float(series.iloc[-1])
            prev = float(series.iloc[-2])
            day_chg = ((latest - prev) / prev) * 100.0

            # Annualized CAGR over dataset duration
            n_days = len(series)
            if n_days > 30 and series.iloc[0] > 0:
                cagr = float(((latest / float(series.iloc[0])) ** (252.0 / n_days) - 1.0) * 100.0)
            else:
                cagr = 15.0

            # Annualized volatility
            returns = series.pct_change().dropna()
            vol = float(returns.std() * np.sqrt(252) * 100.0) if len(returns) > 10 else 18.0

            high_52w = float(series.tail(252).max()) if len(series) >= 20 else latest * 1.15
            low_52w = float(series.tail(252).min()) if len(series) >= 20 else latest * 0.85

            meta = symbol_meta_lookup.get(sym, {})
            name = meta.get("name", sym.replace(".NS", "").replace("_", " ").title())
            sector = meta.get("sector", "Equity Asset")
            pe = meta.get("pe_ratio", 22.0)

            enriched_stocks.append({
                "symbol": sym,
                "name": name,
                "sector": sector,
                "price": round(latest, 2),
                "day_change_pct": round(day_chg, 2),
                "cagr_5y": round(cagr, 1),
                "volatility": round(vol, 1),
                "pe_ratio": pe,
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "source": metadata["source_tag"]
            })

    # Preserve or enrich mutual funds
    mfs = fallback_universe.get("mutual_funds", [])

    return {
        "stocks": enriched_stocks,
        "mutual_funds": mfs,
        "daily_returns_df": daily_returns_df,
        "empirical_cov": empirical_cov,
        "dataset_metadata": metadata,
        "is_kaggle": True,
        "fetch_timestamp": f"Loaded from {metadata['source_tag']} ({metadata.get('rows', 0):,} days)"
    }
