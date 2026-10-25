"""
Agent F: Quantitative Risk Management Agent
Calculates annualized volatility, historical drawdown, Value at Risk (VaR 95%, 98%),
dynamic ATR and structural support stop-loss levels, configurable Risk-Reward targets,
and enforces capital preservation position-sizing rules (max 1%-2% risk budget).
Includes clear disclaimers that stop-losses do not guarantee execution without slippage.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class RiskManagementAgent:
    """
    Evaluates market risk, calculates position sizing, sets stop-loss boundaries,
    and protects investor capital from severe portfolio drawdown.
    """

    def __init__(self, max_risk_budget_pct: float = 1.5):
        self.max_risk_budget_pct = max_risk_budget_pct

    def evaluate_risk(
        self,
        df_ohlcv: pd.DataFrame,
        current_price: float,
        atr_value: float,
        total_capital: float,
        currency_sym: str = "₹",
        risk_reward_ratio: float = 2.0,
        support_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes risk profile, stop-loss, profit targets, position sizes, and limits.
        """
        price = max(float(current_price), 1.0)
        atr = max(float(atr_value), price * 0.01)
        capital = max(float(total_capital), 100.0)

        # 1. Historical Volatility & Drawdown
        if not df_ohlcv.empty and len(df_ohlcv) >= 30:
            returns = df_ohlcv["Close"].pct_change().dropna()
            annualized_vol = float(returns.std() * np.sqrt(252.0) * 100.0)
            rolling_max = df_ohlcv["Close"].cummax()
            drawdowns = (df_ohlcv["Close"] - rolling_max) / rolling_max
            max_drawdown = float(drawdowns.min() * 100.0)
            
            # Parametric Value at Risk (1-Day, 95% and 98%)
            # Z_95 = 1.645, Z_98 = 2.054
            daily_vol = float(returns.std())
            var_95_1d = round(float(1.645 * daily_vol * 100.0), 2)
            var_98_1d = round(float(2.054 * daily_vol * 100.0), 2)
        else:
            annualized_vol = 20.0
            max_drawdown = -15.0
            var_95_1d = 2.1
            var_98_1d = 2.8

        # 2. Dynamic Stop-Loss Levels
        # ATR-based stop-loss: 1.5x ATR
        atr_stop_distance = 1.5 * atr
        atr_stop_price = max(price - atr_stop_distance, price * 0.80)

        # Structural stop: slightly below support level if provided
        if support_price is not None and 0 < support_price < price:
            structural_stop = support_price * 0.985
            # Select tighter or safer stop
            chosen_stop = max(structural_stop, price * 0.85)
            stop_type = "Structural Support Buffer (1.5% below S1)"
        else:
            chosen_stop = atr_stop_price
            stop_type = "Dynamic Volatility Buffer (1.5x ATR)"

        stop_loss_price = round(chosen_stop, 2)
        risk_per_share = round(price - stop_loss_price, 2)
        if risk_per_share <= 0:
            risk_per_share = round(price * 0.05, 2)
            stop_loss_price = round(price * 0.95, 2)

        # 3. Take-Profit Targets based on configurable R:R
        target_price = round(price + (risk_per_share * risk_reward_ratio), 2)
        target_2 = round(price + (risk_per_share * 3.0), 2)  # Extended 1:3 target

        # 4. Position Sizing Rule: Max loss capped at user risk budget (default 1.5% of total capital)
        max_allowed_loss = capital * (self.max_risk_budget_pct / 100.0)
        
        # Quantity based on risk per share
        recommended_shares = int(max_allowed_loss // risk_per_share) if risk_per_share > 0 else 1
        
        # Hard cap: position size cannot exceed 25% of total capital (concentration limit)
        max_affordable_shares = int((capital * 0.25) // price)
        if max_affordable_shares < 1 and capital >= price:
            max_affordable_shares = 1  # Allow single share for micro-investors

        final_shares = min(recommended_shares, max_affordable_shares) if max_affordable_shares > 0 else 0
        final_shares = max(final_shares, 1 if capital >= price else 0)

        position_value = round(final_shares * price, 2)
        actual_max_loss = round(final_shares * risk_per_share, 2)
        potential_profit = round(final_shares * (target_price - price), 2)
        portfolio_exposure_pct = round((position_value / capital) * 100.0, 1)

        # Risk flag checks
        risk_flags = []
        if annualized_vol > 35.0:
            risk_flags.append(f"Elevated asset volatility ({annualized_vol:.1f}% p.a.)")
        if abs(max_drawdown) > 30.0:
            risk_flags.append(f"Deep historical drawdown ({max_drawdown:.1f}%) observed")
        if portfolio_exposure_pct > 20.0:
            risk_flags.append(f"Single position concentration is high ({portfolio_exposure_pct}% of capital)")

        risk_level = "ELEVATED" if len(risk_flags) >= 2 else ("MODERATE" if risk_flags else "CONTROLLED")

        summary = (
            f"Capital Protection Active: For a capital base of {currency_sym} {capital:,.0f}, "
            f"recommended position is {final_shares} shares ({currency_sym} {position_value:,.2f}, representing {portfolio_exposure_pct}% exposure). "
            f"With Stop-Loss at {currency_sym} {stop_loss_price:,.2f}, max planned loss is {currency_sym} {actual_max_loss:,.2f} "
            f"({(actual_max_loss / capital)*100.0:.2f}% of portfolio). Target Price is {currency_sym} {target_price:,.2f} (1:{risk_reward_ratio:.0f} R:R)."
        )

        return {
            "entry_price": round(price, 2),
            "stop_loss_price": stop_loss_price,
            "stop_type": stop_type,
            "risk_per_share": risk_per_share,
            "target_price": target_price,
            "extended_target": target_2,
            "risk_reward_ratio": risk_reward_ratio,
            "recommended_shares": final_shares,
            "position_value": position_value,
            "max_loss_amount": actual_max_loss,
            "potential_profit": potential_profit,
            "portfolio_exposure_pct": portfolio_exposure_pct,
            "annualized_volatility_pct": round(annualized_vol, 1),
            "max_historical_drawdown_pct": round(max_drawdown, 1),
            "var_95_1d_pct": var_95_1d,
            "var_98_1d_pct": var_98_1d,
            "risk_level": risk_level,
            "risk_flags": risk_flags,
            "summary": summary,
            "disclaimer": "⚠️ A stop-loss order does not guarantee execution at the exact specified price during market gaps, illiquidity, or severe volatility slippage."
        }
