"""
Simulation Engine:
Calculates Best, Average, and Worst Case Wealth Projections using
98% Statistical Confidence Monte Carlo Simulation and Historical Readings.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def run_monte_carlo_simulation(
    initial_lump_sum: float,
    monthly_sip: float,
    expected_cagr_pct: float,
    volatility_pct: float,
    horizon_years: int = 10,
    num_simulations: int = 5000,
    confidence_level: float = 0.98
) -> Dict[str, Any]:
    """
    Runs a 98% statistical confidence Monte Carlo simulation over horizon_years.
    Models combined lump-sum and recurring monthly SIP contributions with stochastic drift.
    """
    mu_annual = expected_cagr_pct / 100.0
    sigma_annual = volatility_pct / 100.0

    # Monthly parameters
    dt = 1.0 / 12.0
    mu_monthly = (1.0 + mu_annual) ** dt - 1.0
    sigma_monthly = sigma_annual * np.sqrt(dt)

    total_months = horizon_years * 12
    np.random.seed(42)  # Deterministic seed for reproducible and stable UI runs

    # Generate random shocks for all paths: shape (num_simulations, total_months)
    shocks = np.random.normal(0, 1, size=(num_simulations, total_months))

    # Monthly return multipliers using Geometric Brownian Motion
    monthly_growth = np.exp((mu_monthly - 0.5 * (sigma_monthly ** 2)) + sigma_monthly * shocks)

    # Simulation array to track wealth path
    portfolio_paths = np.zeros((num_simulations, total_months + 1))
    portfolio_paths[:, 0] = initial_lump_sum

    invested_capital = np.zeros(total_months + 1)
    invested_capital[0] = initial_lump_sum

    for m in range(1, total_months + 1):
        # Existing wealth grows by monthly factor, plus monthly SIP addition
        portfolio_paths[:, m] = portfolio_paths[:, m - 1] * monthly_growth[:, m - 1] + monthly_sip
        invested_capital[m] = invested_capital[m - 1] + monthly_sip

    # Calculate percentiles at each yearly milestone
    # For a 98% confidence interval:
    # Lower bound = (1 - 0.98) / 2 = 1.0% percentile (or 2.0% percentile for conservative worst-case)
    # Median = 50.0% percentile
    # Upper bound = 100 - 1.0% = 99.0% percentile (or 98.0% percentile)
    alpha = (1.0 - confidence_level) / 2.0  # 0.01 for 98%
    p_lower = alpha * 100.0                # 1.0%
    p_upper = (1.0 - alpha) * 100.0        # 99.0%

    yearly_data = []
    for y in range(horizon_years + 1):
        m_idx = y * 12
        values_at_m = portfolio_paths[:, m_idx]

        worst_val = float(np.percentile(values_at_m, p_lower))
        avg_val = float(np.percentile(values_at_m, 50.0))
        best_val = float(np.percentile(values_at_m, p_upper))
        capital_val = float(invested_capital[m_idx])

        yearly_data.append({
            "Year": y,
            "Month": m_idx,
            "Invested_Capital": round(capital_val, 2),
            "Worst_Case": round(worst_val, 2),
            "Average_Case": round(avg_val, 2),
            "Best_Case": round(best_val, 2)
        })

    df_projections = pd.DataFrame(yearly_data)

    final_row = df_projections.iloc[-1]
    total_invested = final_row["Invested_Capital"]

    # 1-Year Value at Risk (VaR) at 98% confidence
    # VaR_98 = -(mu - Z_0.98 * sigma)
    # Z for 98% is approximately 2.054
    z_score_98 = 2.0537
    var_1y_pct = round(max((z_score_98 * sigma_annual - mu_annual) * 100.0, 0.0), 2)
    max_drawdown_estimated = round(min(volatility_pct * 1.8, 48.0), 1)

    return {
        "df_projections": df_projections,
        "total_invested": total_invested,
        "final_worst": final_row["Worst_Case"],
        "final_avg": final_row["Average_Case"],
        "final_best": final_row["Best_Case"],
        "worst_return_pct": round(((final_row["Worst_Case"] - total_invested) / total_invested) * 100.0, 1),
        "avg_return_pct": round(((final_row["Average_Case"] - total_invested) / total_invested) * 100.0, 1),
        "best_return_pct": round(((final_row["Best_Case"] - total_invested) / total_invested) * 100.0, 1),
        "var_1y_98_pct": var_1y_pct,
        "max_drawdown_est_pct": max_drawdown_estimated,
        "confidence_level_pct": int(confidence_level * 100),
        "num_simulations": num_simulations
    }
