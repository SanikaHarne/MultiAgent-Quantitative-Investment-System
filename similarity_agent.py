"""
Agent D: Historical Similarity Agent
Discovers comparable historical market regimes and patterns using standardized
multi-factor vectors (returns, trend distances, RSI, MACD, volatility, volume, drawdown).
Enforces minimum time gaps between analogues to prevent overlapping observation bias.
Measures empirical forward returns at 1-day, 5-day, and 20-day horizons.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def extract_feature_matrix(df_ohlcv: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes strictly backward-looking feature vectors for every trading day.
    No future data is ever used.
    """
    df = df_ohlcv.copy()
    close = df["Close"]

    # 1. Multi-window percentage returns
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)

    # 2. Distance from Moving Averages
    sma_20 = close.rolling(window=20, min_periods=5).mean()
    sma_50 = close.rolling(window=50, min_periods=10).mean()
    sma_200 = close.rolling(window=min(200, len(df)), min_periods=20).mean()

    df["dist_sma20"] = (close / (sma_20 + 1e-9)) - 1.0
    df["dist_sma50"] = (close / (sma_50 + 1e-9)) - 1.0
    df["dist_sma200"] = (close / (sma_200 + 1e-9)) - 1.0

    # 3. RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 4. MACD Histogram Normalized
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (macd - macd_signal) / (close + 1e-9)

    # 5. Volatility Regime (20-day annualized realized vol)
    df["volatility_20d"] = df["ret_1d"].rolling(window=20).std() * np.sqrt(252.0)

    # 6. Volume Change Ratio
    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(window=20).mean()
        df["vol_ratio"] = df["Volume"] / (vol_ma + 1e-9)
    else:
        df["vol_ratio"] = 1.0

    # 7. Drawdown from 52-Week Rolling High
    rolling_high_252 = close.rolling(window=min(252, len(df)), min_periods=20).max()
    df["drawdown_52w"] = (close / (rolling_high_252 + 1e-9)) - 1.0

    # 8. Relative Strength vs Benchmark
    if benchmark_df is not None and not benchmark_df.empty:
        bench_ret_20 = benchmark_df["Close"].pct_change(20).reindex(df.index, method="ffill")
        df["rel_strength_20d"] = df["ret_20d"] - bench_ret_20
    else:
        df["rel_strength_20d"] = 0.0

    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "dist_sma20", "dist_sma50", "dist_sma200",
        "rsi_14", "macd_hist_norm", "volatility_20d",
        "vol_ratio", "drawdown_52w", "rel_strength_20d"
    ]

    clean_df = df.dropna(subset=feature_cols).copy()
    return clean_df, feature_cols


class HistoricalSimilarityAgent:
    """
    Finds verified historical analogues, deduplicates overlapping windows,
    and calculates forward statistical outcome distributions.
    """

    def __init__(self, top_k: int = 15, min_gap_days: int = 15):
        self.top_k = top_k
        self.min_gap_days = min_gap_days

    def analyze(self, df_ohlcv: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Runs full similarity matching pipeline on the provided historical series.
        """
        if df_ohlcv.empty or len(df_ohlcv) < 150:
            return {
                "status": "INSUFFICIENT_DATA",
                "summary": "At least 150 daily bars required to construct historical similarity matrices.",
                "analogues_found": 0,
                "horizons": {},
                "analogues": []
            }

        feat_df, feature_cols = extract_feature_matrix(df_ohlcv, benchmark_df)
        if len(feat_df) < 100:
            return {
                "status": "INSUFFICIENT_DATA",
                "summary": "Insufficient valid feature rows after warmup period.",
                "analogues_found": 0,
                "horizons": {},
                "analogues": []
            }

        # The latest row is the CURRENT market state
        current_state = feat_df.iloc[-1]
        current_vector = current_state[feature_cols].values.reshape(1, -1)

        # Candidate historical search space: exclude the most recent 30 trading days to prevent self-matching
        search_space = feat_df.iloc[:-30].copy()
        if len(search_space) < 50:
            return {
                "status": "INSUFFICIENT_DATA",
                "summary": "Historical search space too short to find non-overlapping analogues.",
                "analogues_found": 0,
                "horizons": {},
                "analogues": []
            }

        # Standardize features using search space statistics only (Z-score normalization)
        mean_vec = search_space[feature_cols].mean().values
        std_vec = search_space[feature_cols].std().values + 1e-9

        norm_current = (current_vector - mean_vec) / std_vec
        norm_history = (search_space[feature_cols].values - mean_vec) / std_vec

        # Cosine distance: similarity = 1 - cosine_distance
        dists = cdist(norm_current, norm_history, metric="cosine")[0]
        # Handle nan in cosine distance
        dists = np.nan_to_num(dists, nan=2.0)
        similarity_scores = 1.0 - (dists / 2.0)  # Maps cosine distance [0, 2] to [0.0, 1.0]

        search_space["similarity"] = similarity_scores
        sorted_candidates = search_space.sort_values(by="similarity", ascending=False)

        # Select top non-overlapping analogues
        selected_analogues = []
        selected_indices = []

        all_dates = list(df_ohlcv.index)
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        for date_val, row in sorted_candidates.iterrows():
            idx = date_to_idx.get(date_val)
            if idx is None:
                continue

            # Ensure minimum gap from previously selected analogues
            if any(abs(idx - prev_idx) < self.min_gap_days for prev_idx in selected_indices):
                continue

            # Ensure forward data exists for measuring 1d, 5d, 20d outcomes
            if idx + 20 >= len(all_dates):
                continue

            selected_indices.append(idx)
            selected_analogues.append({
                "date": date_val.strftime("%Y-%m-%d"),
                "date_dt": date_val,
                "idx": idx,
                "similarity": round(float(row["similarity"]) * 100.0, 1),
                "rsi": round(float(row["rsi_14"]), 1),
                "ret_20d": round(float(row["ret_20d"]) * 100.0, 2)
            })

            if len(selected_analogues) >= self.top_k:
                break

        if len(selected_analogues) < 3:
            return {
                "status": "LOW_MATCHES",
                "summary": f"Only {len(selected_analogues)} distinct historical analogues met the similarity threshold.",
                "analogues_found": len(selected_analogues),
                "horizons": {},
                "analogues": selected_analogues
            }

        # Measure forward subsequent returns across horizons
        horizons_config = [
            {"key": "1d", "name": "Next 1 Trading Day", "bars": 1},
            {"key": "5d", "name": "Next 5 Trading Days (1-Week)", "bars": 5},
            {"key": "20d", "name": "Next 20 Trading Days (1-Month)", "bars": 20}
        ]

        horizons_summary = {}

        for h in horizons_config:
            bars = h["bars"]
            fwd_returns = []

            for a in selected_analogues:
                idx = a["idx"]
                p_entry = float(df_ohlcv["Close"].iloc[idx])
                p_exit = float(df_ohlcv["Close"].iloc[idx + bars])
                ret_pct = ((p_exit - p_entry) / p_entry) * 100.0
                fwd_returns.append(ret_pct)
                a[f"fwd_{h['key']}_ret"] = round(ret_pct, 2)

            fwd_arr = np.array(fwd_returns)
            pos_count = int(np.sum(fwd_arr > 0))
            neg_count = int(np.sum(fwd_arr < 0))
            flat_count = len(fwd_arr) - pos_count - neg_count

            win_rate = round((pos_count / len(fwd_arr)) * 100.0, 1)
            loss_rate = round((neg_count / len(fwd_arr)) * 100.0, 1)
            med_ret = round(float(np.median(fwd_arr)), 2)
            mean_ret = round(float(np.mean(fwd_arr)), 2)
            p25 = round(float(np.percentile(fwd_arr, 25)), 2)
            p75 = round(float(np.percentile(fwd_arr, 75)), 2)
            worst = round(float(np.min(fwd_arr)), 2)
            best = round(float(np.max(fwd_arr)), 2)

            horizons_summary[h["key"]] = {
                "name": h["name"],
                "horizon_bars": bars,
                "sample_count": len(fwd_arr),
                "win_rate_pct": win_rate,
                "loss_rate_pct": loss_rate,
                "median_return_pct": med_ret,
                "mean_return_pct": mean_ret,
                "p25_pct": p25,
                "p75_pct": p75,
                "worst_case_pct": worst,
                "best_case_pct": best,
                "returns_distribution": [round(float(v), 2) for v in fwd_arr]
            }

        avg_similarity = round(float(np.mean([a["similarity"] for a in selected_analogues])), 1)

        summary_text = (
            f"Found {len(selected_analogues)} distinct historical analogues with average similarity of {avg_similarity}%. "
            f"Over the subsequent 20 trading days, {horizons_summary['20d']['win_rate_pct']}% of comparable episodes produced "
            f"positive returns with a median performance of {horizons_summary['20d']['median_return_pct']:+.1f}% "
            f"(25th-75th quartile range: [{horizons_summary['20d']['p25_pct']:+.1f}%, {horizons_summary['20d']['p75_pct']:+.1f}%])."
        )

        return {
            "status": "SUCCESS",
            "analogues_found": len(selected_analogues),
            "avg_similarity_pct": avg_similarity,
            "horizons": horizons_summary,
            "analogues": selected_analogues,
            "summary": summary_text,
            "disclaimer": "⚠️ Historical pattern matching is empirical and probabilistic. Past market analogues do not guarantee future performance."
        }
