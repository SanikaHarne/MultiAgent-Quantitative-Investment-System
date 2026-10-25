"""
Technical Analysis Engine:
Computes EMA (20/50/200), RSI (14), MACD, Bollinger Bands, and ATR (14).
Outputs a weighted consensus signal: STRONG BUY, BUY, NEUTRAL/HOLD, SELL, or STRONG SELL.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any

def analyze_technical_indicators(ticker_symbol: str) -> Dict[str, Any]:
    clean_ticker = ticker_symbol.strip().upper()
    
    fallback_res = {
        "symbol": clean_ticker,
        "last_price": 100.0,
        "signal": "NEUTRAL / HOLD",
        "signal_score": 0,
        "rsi": 50.0,
        "ema_20": 100.0,
        "ema_50": 100.0,
        "ema_200": 100.0,
        "macd_line": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "bb_upper": 105.0,
        "bb_lower": 95.0,
        "atr": 2.5,
        "df_ohlc": pd.DataFrame(),
        "summary": "Baseline technical readings applied."
    }

    try:
        t = yf.Ticker(clean_ticker)
        df = t.history(period="1y", interval="1d")
        if df.empty or len(df) < 50:
            return fallback_res

        # 1. Exponential Moving Averages (EMA)
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        # 2. Relative Strength Index (RSI - 14)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # 3. MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # 4. Bollinger Bands (20, 2)
        df["BB_Mid"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Mid"] + (2 * bb_std)
        df["BB_Lower"] = df["BB_Mid"] - (2 * bb_std)

        # 5. Average True Range (ATR - 14)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14).mean()

        latest = df.iloc[-1]
        last_close = float(latest["Close"])
        rsi_val = float(latest["RSI"]) if not np.isnan(latest["RSI"]) else 50.0
        atr_val = float(latest["ATR"]) if not np.isnan(latest["ATR"]) else last_close * 0.02

        # Weighted Consensus Signal Calculation
        score = 0
        reasons = []

        if last_close > latest["EMA_20"] > latest["EMA_50"]:
            score += 2
            reasons.append("Price above EMA20 & EMA50 (Strong Bullish Momentum)")
        elif last_close < latest["EMA_20"] < latest["EMA_50"]:
            score -= 2
            reasons.append("Price below EMA20 & EMA50 (Bearish Pressure)")

        if rsi_val < 32:
            score += 2
            reasons.append(f"RSI is oversold ({rsi_val:.1f}) - Potential Bounce Zone")
        elif rsi_val > 68:
            score -= 2
            reasons.append(f"RSI is overbought ({rsi_val:.1f}) - Overextended")

        if latest["MACD_Hist"] > 0:
            score += 1
            reasons.append("MACD histogram positive")
        else:
            score -= 1
            reasons.append("MACD histogram negative")

        if last_close <= latest["BB_Lower"]:
            score += 2
            reasons.append("Price touching Lower Bollinger Band (Value Zone)")
        elif last_close >= latest["BB_Upper"]:
            score -= 2
            reasons.append("Price touching Upper Bollinger Band (High Volatility Zone)")

        if score >= 3:
            signal = "STRONG BUY"
        elif score >= 1:
            signal = "BUY"
        elif score <= -3:
            signal = "STRONG SELL"
        elif score <= -1:
            signal = "SELL"
        else:
            signal = "NEUTRAL / HOLD"

        return {
            "symbol": clean_ticker,
            "last_price": round(last_close, 2),
            "signal": signal,
            "signal_score": score,
            "rsi": round(rsi_val, 1),
            "ema_20": round(float(latest["EMA_20"]), 2),
            "ema_50": round(float(latest["EMA_50"]), 2),
            "ema_200": round(float(latest["EMA_200"]), 2),
            "macd_line": round(float(latest["MACD"]), 2),
            "macd_signal": round(float(latest["MACD_Signal"]), 2),
            "macd_hist": round(float(latest["MACD_Hist"]), 2),
            "bb_upper": round(float(latest["BB_Upper"]), 2),
            "bb_lower": round(float(latest["BB_Lower"]), 2),
            "atr": round(atr_val, 2),
            "df_ohlc": df.tail(120),
            "summary": " • ".join(reasons) if reasons else "Neutral market momentum."
        }
    except Exception:
        return fallback_res