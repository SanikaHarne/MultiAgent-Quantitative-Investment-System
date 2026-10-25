"""
Agent C: Technical Analysis Agent
Computes comprehensive indicators: EMAs (20/50/200), RSI (14), MACD, Bollinger Bands, ATR (14),
pivot support/resistance levels, and relative strength against benchmark indices.
Evaluates trend regimes and produces an evidence-based technical assessment.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class TechnicalAnalysisAgent:
    """
    Evaluates price trend, momentum, volatility, support/resistance,
    and relative strength without making deterministic profit promises.
    """

    def __init__(self, benchmark_df: Optional[pd.DataFrame] = None):
        self.benchmark_df = benchmark_df

    def analyze(self, df_ohlcv: pd.DataFrame, ticker: str = "") -> Dict[str, Any]:
        """
        Runs complete technical indicator pipeline on OHLCV DataFrame.
        """
        if df_ohlcv.empty or len(df_ohlcv) < 30:
            return {
                "symbol": ticker,
                "status": "INSUFFICIENT_DATA",
                "summary": "Insufficient candle history to compute full technical indicators.",
                "stance": "NEUTRAL / INSUFFICIENT DATA",
                "indicators": {},
                "df_indicators": pd.DataFrame()
            }

        df = df_ohlcv.copy()

        # 1. Exponential Moving Averages (EMA 20, 50, 200)
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        if len(df) >= 200:
            df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
        else:
            df["EMA_200"] = df["Close"].ewm(span=len(df), adjust=False).mean()

        # 2. Relative Strength Index (RSI 14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        # 3. MACD (12, 26, 9)
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # 4. Bollinger Bands (20, 2 std)
        df["BB_Mid"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Mid"] + (2.0 * bb_std)
        df["BB_Lower"] = df["BB_Mid"] - (2.0 * bb_std)
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Mid"] + 1e-9)

        # 5. Average True Range (ATR 14)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14).mean()

        # 6. Volume Moving Average
        if "Volume" in df.columns:
            df["Vol_MA20"] = df["Volume"].rolling(window=20).mean()
            df["Vol_Ratio"] = df["Volume"] / (df["Vol_MA20"] + 1e-9)
        else:
            df["Vol_Ratio"] = 1.0

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        last_price = float(latest["Close"])
        ema_20 = float(latest["EMA_20"])
        ema_50 = float(latest["EMA_50"])
        ema_200 = float(latest["EMA_200"])
        rsi_val = float(latest["RSI"]) if not np.isnan(latest["RSI"]) else 50.0
        macd_line = float(latest["MACD"])
        macd_signal = float(latest["MACD_Signal"])
        macd_hist = float(latest["MACD_Hist"])
        bb_upper = float(latest["BB_Upper"]) if not np.isnan(latest["BB_Upper"]) else last_price * 1.05
        bb_lower = float(latest["BB_Lower"]) if not np.isnan(latest["BB_Lower"]) else last_price * 0.95
        atr_val = float(latest["ATR"]) if not np.isnan(latest["ATR"]) else last_price * 0.02
        vol_ratio = float(latest["Vol_Ratio"]) if not np.isnan(latest["Vol_Ratio"]) else 1.0

        # 7. Classical Pivot Points (Floor Pivots)
        high_p = float(latest["High"])
        low_p = float(latest["Low"])
        close_p = last_price
        pivot = (high_p + low_p + close_p) / 3.0
        r1 = (2.0 * pivot) - low_p
        s1 = (2.0 * pivot) - high_p
        r2 = pivot + (high_p - low_p)
        s2 = pivot - (high_p - low_p)

        # 8. Relative Strength vs Benchmark Index
        rel_strength_20d = 0.0
        rel_strength_summary = "Benchmark data not attached."
        if self.benchmark_df is not None and not self.benchmark_df.empty and len(self.benchmark_df) >= 20:
            stock_ret_20d = (last_price / float(df["Close"].iloc[-20]) - 1.0) * 100.0
            bench_latest = float(self.benchmark_df["Close"].iloc[-1])
            bench_20d_ago = float(self.benchmark_df["Close"].iloc[-20])
            bench_ret_20d = (bench_latest / bench_20d_ago - 1.0) * 100.0
            rel_strength_20d = round(stock_ret_20d - bench_ret_20d, 2)
            if rel_strength_20d > 2.0:
                rel_strength_summary = f"Outperforming benchmark by +{rel_strength_20d:.1f}% over 20 days."
            elif rel_strength_20d < -2.0:
                rel_strength_summary = f"Underperforming benchmark by {rel_strength_20d:.1f}% over 20 days."
            else:
                rel_strength_summary = f"Tracking benchmark closely ({rel_strength_20d:+.1f}% vs index)."

        # Evidence accumulation
        evidence_points = []
        score = 0

        # Moving Average Alignment
        if last_price > ema_20 > ema_50:
            score += 2
            evidence_points.append("Price trades above ascending EMA20 and EMA50 (Bullish Trend)")
        elif last_price < ema_20 < ema_50:
            score -= 2
            evidence_points.append("Price trades below declining EMA20 and EMA50 (Bearish Trend)")
        else:
            evidence_points.append("Moving averages tangled (Consolidation / Rangebound)")

        if last_price > ema_200:
            score += 1
            evidence_points.append("Holding above long-term 200 EMA support")
        else:
            score -= 1
            evidence_points.append("Trading below long-term 200 EMA barrier")

        # RSI Momentum
        if rsi_val < 32:
            score += 2
            evidence_points.append(f"RSI oversold ({rsi_val:.1f}) — potential mean-reversion bounce")
        elif rsi_val > 68:
            score -= 2
            evidence_points.append(f"RSI overbought ({rsi_val:.1f}) — extended upside resistance")
        else:
            evidence_points.append(f"RSI in balanced zone ({rsi_val:.1f})")

        # MACD
        if macd_hist > 0 and float(prev["MACD_Hist"]) <= 0:
            score += 2
            evidence_points.append("Bullish MACD centerline histogram crossover")
        elif macd_hist > 0:
            score += 1
            evidence_points.append("MACD momentum positive")
        else:
            score -= 1
            evidence_points.append("MACD momentum negative")

        # Bollinger Bands
        if last_price <= bb_lower:
            score += 1
            evidence_points.append("Testing lower Bollinger Band boundary")
        elif last_price >= bb_upper:
            score -= 1
            evidence_points.append("Stretching above upper Bollinger Band")

        # Volume confirmation
        if vol_ratio > 1.3:
            evidence_points.append(f"Elevated trading volume ({vol_ratio:.1f}x 20-day average)")

        # Overall stance
        if score >= 4:
            stance = "STRONGLY BULLISH"
        elif score >= 2:
            stance = "MODERATELY BULLISH"
        elif score <= -4:
            stance = "STRONGLY BEARISH"
        elif score <= -2:
            stance = "MODERATELY BEARISH"
        else:
            stance = "NEUTRAL / CONSOLIDATION"

        return {
            "symbol": ticker,
            "status": "SUCCESS",
            "last_price": round(last_price, 2),
            "stance": stance,
            "composite_score": score,
            "indicators": {
                "rsi": round(rsi_val, 1),
                "ema_20": round(ema_20, 2),
                "ema_50": round(ema_50, 2),
                "ema_200": round(ema_200, 2),
                "macd_line": round(macd_line, 2),
                "macd_signal": round(macd_signal, 2),
                "macd_hist": round(macd_hist, 2),
                "bb_upper": round(bb_upper, 2),
                "bb_lower": round(bb_lower, 2),
                "atr": round(atr_val, 2),
                "volume_ratio": round(vol_ratio, 2),
                "pivot": round(pivot, 2),
                "resistance_1": round(r1, 2),
                "support_1": round(s1, 2),
                "resistance_2": round(r2, 2),
                "support_2": round(s2, 2),
                "rel_strength_20d": rel_strength_20d
            },
            "evidence": evidence_points,
            "rel_strength_summary": rel_strength_summary,
            "summary": " • ".join(evidence_points[:4]),
            "df_indicators": df.tail(120)
        }
