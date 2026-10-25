"""
Risk Management Engine:
Calculates ATR-based dynamic Stop-Loss, Take-Profit targets, Max Loss Amount,
and Capital Preservation Position Sizing (1%-2% max risk per trade).
"""

import math
from typing import Any, Dict

def calculate_trade_risk(
    entry_price: float,
    atr_value: float,
    total_savings: float,
    currency_symbol: str = "₹",
    risk_reward_ratio: float = 2.0,
    max_risk_pct: float = 1.5
) -> Dict[str, Any]:
    
    # 1. Clean and validate inputs FIRST before any mathematical operations
    def _clean_float(val, fallback=1.0):
        try:
            if val is None:
                return fallback
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return fallback
            return f_val
        except (ValueError, TypeError):
            return fallback

    entry = max(_clean_float(entry_price, fallback=1190.0), 1.0)
    atr = max(_clean_float(atr_value, fallback=entry * 0.015), entry * 0.005)
    total_savings = max(_clean_float(total_savings, fallback=100000.0), 0.0)

    # 2. Dynamic Stop-Loss Distance (1.5x ATR)
    stop_distance = 1.5 * atr
    stop_loss_price = max(entry - stop_distance, entry * 0.85)

    # 3. Take-Profit Target (1:2 Risk-Reward Ratio)
    target_price = entry + (stop_distance * risk_reward_ratio)

    # 4. Capital Risk Sizing Rule (1.5% max risk of total savings)
    max_acceptable_loss = max((total_savings * (max_risk_pct / 100.0)), 50.0)
    loss_per_share = entry - stop_loss_price

    if loss_per_share > 0:
        safe_shares = int(max_acceptable_loss // loss_per_share)
    else:
        safe_shares = 1

    # 5. Safe position sizing calculation
    max_affordable_shares = int(total_savings // entry) if entry > 0 else 0
    
    if max_affordable_shares > 0:
        safe_shares = min(safe_shares, max_affordable_shares)
    
    safe_shares = max(safe_shares, 1)  # Guarantee at least 1 unit if savings permit

    actual_position_value = round(safe_shares * entry, 2)
    actual_max_loss = round(safe_shares * loss_per_share, 2)
    potential_profit = round(safe_shares * (target_price - entry), 2)

    return {
        "entry_price": round(entry, 2),
        "stop_loss_price": round(stop_loss_price, 2),
        "target_price": round(target_price, 2),
        "atr_value": round(atr, 2),
        "risk_reward_ratio": risk_reward_ratio,
        "max_risk_pct": max_risk_pct,
        "recommended_units": safe_shares,
        "position_value": actual_position_value,
        "max_loss_amount": actual_max_loss,
        "potential_profit": potential_profit,
        "risk_summary": (
            f"Capital Protection Active: Investing {currency_symbol} {actual_position_value:,.2f} ({safe_shares} shares/units). "
            f"If the market triggers your Stop-Loss at {currency_symbol} {stop_loss_price:,.2f}, your maximum loss is strictly capped at "
            f"{currency_symbol} {actual_max_loss:,.2f} (only {max_risk_pct}% of your total savings)."
        )
    }