"""
Options Risk Simulator:
Models Black-Scholes Call/Put theoretical pricing, Option Greeks,
payoff matrices (-10% to +10%), and beginner safety advice.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any

def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> Dict[str, float]:
    S = max(float(S), 1.0)
    K = max(float(K), 1.0)
    T = max(float(T) / 365.0, 0.001)
    r = max(float(r), 0.01)
    sigma = max(float(sigma), 0.05)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    
    if option_type.lower() == "call":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0

    return {
        "price": round(float(price), 2),
        "delta": round(float(delta), 3),
        "gamma": round(float(gamma), 4),
        "theta": round(float(theta), 3),
        "vega": round(float(vega), 3)
    }

def simulate_options_payoff(
    stock_price: float,
    strike_price: float,
    days_to_expiry: int = 30,
    volatility_pct: float = 20.0,
    risk_free_rate: float = 0.065,
    lot_size: int = 100
) -> Dict[str, Any]:
    S0 = float(stock_price)
    K = float(strike_price)
    sigma = float(volatility_pct) / 100.0
    
    call_bs = black_scholes_greeks(S0, K, days_to_expiry, risk_free_rate, sigma, "call")
    put_bs = black_scholes_greeks(S0, K, days_to_expiry, risk_free_rate, sigma, "put")

    call_premium = max(call_bs["price"], 1.0)
    put_premium = max(put_bs["price"], 1.0)

    price_swings_pct = np.linspace(-10.0, 10.0, 9)
    payoff_rows = []

    for pct in price_swings_pct:
        sim_S = S0 * (1.0 + (pct / 100.0))
        call_intrinsic = max(sim_S - K, 0.0)
        put_intrinsic = max(K - sim_S, 0.0)

        call_pnl = (call_intrinsic - call_premium) * lot_size
        put_pnl = (put_intrinsic - put_premium) * lot_size

        payoff_rows.append({
            "Stock Change %": f"{pct:+.1f}%",
            "Simulated Price": round(sim_S, 2),
            "Call PnL": round(call_pnl, 2),
            "Put PnL": round(put_pnl, 2)
        })

    return {
        "call_greeks": call_bs,
        "put_greeks": put_bs,
        "call_premium": call_premium,
        "put_premium": put_premium,
        "call_max_loss": round(call_premium * lot_size, 2),
        "put_max_loss": round(put_premium * lot_size, 2),
        "payoff_df": pd.DataFrame(payoff_rows),
        "advice": "⚠️ Options carry significant risk from rapid time decay (Theta). Beginners saving capital should stick to Direct Equity or Index ETFs."
    }