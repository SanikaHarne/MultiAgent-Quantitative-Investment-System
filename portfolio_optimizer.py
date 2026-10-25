"""
Modern Portfolio Theory (MPT) Optimization & Asset Allocation Engine.
Implements Markowitz Mean-Variance Optimization, Dynamic Sharpe Ratio Maximization via SciPy,
Global Minimum Variance (GMV) Portfolio, and Efficient Frontier Generation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import ALLOCATION_PROFILES

# Default Risk-Free Rates (10-Year Sovereign Yield Proxies)
RISK_FREE_RATES = {
    "INR": 0.065,  # 6.50% Indian 10Y G-Sec
    "USD": 0.0425  # 4.25% US 10Y Treasury
}


def build_synthetic_covariance(
    volatilities: np.ndarray,
    avg_correlation: float = 0.35
) -> np.ndarray:
    """
    Constructs a valid positive semi-definite covariance matrix
    from asset annualized volatilities and empirical sector correlation.
    Sigma_ij = rho_ij * sigma_i * sigma_j
    """
    n = len(volatilities)
    corr = np.full((n, n), avg_correlation)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(volatilities, volatilities) * corr
    return cov


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> Tuple[float, float, float]:
    """
    Calculates Portfolio Expected Return, Portfolio Volatility, and Sharpe Ratio.
    """
    ret = np.sum(expected_returns * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-6 else 0.0
    return ret, vol, sharpe


def optimize_max_sharpe(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> Tuple[np.ndarray, float, float, float]:
    """
    Solves for Maximum Sharpe Ratio (Tangency Portfolio) using SciPy SLSQP.
    min -Sharpe(w) s.t. sum(w) = 1, 0 <= w_i <= 1
    """
    n = len(expected_returns)
    init_weights = np.ones(n) / n
    bounds = tuple((0.02, 0.45) for _ in range(n))  # Diversification constraints (2% min, 45% max)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def neg_sharpe(w):
        _, vol, sharpe = portfolio_performance(w, expected_returns, cov_matrix, risk_free_rate)
        return -sharpe

    res = minimize(
        neg_sharpe,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    opt_weights = res.x if res.success else init_weights
    opt_weights = opt_weights / np.sum(opt_weights)
    ret, vol, sharpe = portfolio_performance(opt_weights, expected_returns, cov_matrix, risk_free_rate)
    return opt_weights, ret, vol, sharpe


def optimize_min_variance(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> Tuple[np.ndarray, float, float, float]:
    """
    Solves for Global Minimum Variance (GMV) Portfolio.
    min Var(w) s.t. sum(w) = 1, 0 <= w_i <= 1
    """
    n = len(expected_returns)
    init_weights = np.ones(n) / n
    bounds = tuple((0.02, 0.50) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    res = minimize(
        portfolio_vol,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    opt_weights = res.x if res.success else init_weights
    opt_weights = opt_weights / np.sum(opt_weights)
    ret, vol, sharpe = portfolio_performance(opt_weights, expected_returns, cov_matrix, risk_free_rate)
    return opt_weights, ret, vol, sharpe


def generate_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    num_points: int = 40
) -> Dict[str, Any]:
    """
    Traces the Markowitz Efficient Frontier curve and simulates a Monte Carlo
    random portfolio cloud to visualize the investment opportunity set.
    """
    n = len(expected_returns)
    _, _, min_vol, _ = optimize_min_variance(expected_returns, cov_matrix, risk_free_rate)
    target_returns = np.linspace(np.min(expected_returns), np.max(expected_returns), num_points)

    frontier_vols = []
    frontier_rets = []
    frontier_sharpes = []

    bounds = tuple((0.0, 1.0) for _ in range(n))

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: np.sum(w * expected_returns) - target}
        )
        res = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if res.success:
            vol = res.fun
            ret = target
            sharpe = (ret - risk_free_rate) / vol if vol > 1e-6 else 0.0
            frontier_vols.append(vol * 100.0)
            frontier_rets.append(ret * 100.0)
            frontier_sharpes.append(sharpe)

    # Simulate 1,200 random portfolios for the feasible cloud
    np.random.seed(42)
    random_weights = np.random.dirichlet(np.ones(n), size=1200)
    rand_rets = np.sum(random_weights * expected_returns, axis=1) * 100.0
    rand_vols = np.sqrt(np.einsum("ij,jk,ik->i", random_weights, cov_matrix, random_weights)) * 100.0
    rand_sharpes = (rand_rets / 100.0 - risk_free_rate) / (rand_vols / 100.0)

    return {
        "frontier_vol": frontier_vols,
        "frontier_ret": frontier_rets,
        "frontier_sharpe": frontier_sharpes,
        "random_vols": rand_vols.tolist(),
        "random_rets": rand_rets.tolist(),
        "random_sharpes": rand_sharpes.tolist()
    }


def optimize_portfolio(
    amount: float,
    currency: str,
    risk_profile: str,
    market: str,
    universe: Dict[str, Any],
    sentiment_adjustment: float = 0.0
) -> Dict[str, Any]:
    """
    Main Multi-Asset Allocation and Quantitative Optimization Entry Point.
    Blends Modern Portfolio Theory (MPT) with real-world integer share unit constraints,
    micro-SIP allocation rules, and news sentiment CAGR tilting.
    """
    stocks_available = universe.get("stocks", [])
    mfs_available = universe.get("mutual_funds", [])
    currency_code = "INR" if ("INR" in currency or currency == "₹") else "USD"
    rf_rate = RISK_FREE_RATES.get(currency_code, 0.05)

    micro_threshold = 2000.0 if currency_code == "INR" else 75.0

    # ------------------ CASE 1: Student Micro-Investment Route (₹100 to ₹2,000) ------------------
    if amount < micro_threshold:
        primary_mf = mfs_available[0] if mfs_available else {}
        base_cagr = primary_mf.get("cagr_5y", 15.0) + sentiment_adjustment
        base_vol = primary_mf.get("volatility", 14.0)

        # Check for affordable micro-stocks / ETFs (e.g. Gold BeES @ ₹68, Tata Steel @ ₹160, Nifty BeES @ ₹285)
        affordable_stocks = [s for s in stocks_available if s.get("price", 9999.0) <= (amount * 0.70)]
        stock_picks = []
        stocks_total = 0.0

        if affordable_stocks and amount >= 140.0:
            top_micro_stock = affordable_stocks[0]
            qty = int((amount * 0.45) // top_micro_stock["price"]) or 1
            cost = round(qty * top_micro_stock["price"], 2)
            if cost <= amount:
                stock_picks.append({
                    "symbol": top_micro_stock["symbol"],
                    "name": top_micro_stock["name"],
                    "price": top_micro_stock["price"],
                    "quantity": qty,
                    "allocated_amount": cost,
                    "sector": top_micro_stock.get("sector", "Micro Equity"),
                    "cagr_5y": top_micro_stock.get("cagr_5y", 15.0),
                    "volatility": top_micro_stock.get("volatility", 15.0),
                    "pe_ratio": top_micro_stock.get("pe_ratio", 20.0),
                    "mpt_weight_pct": round((cost / amount) * 100, 1),
                    "day_change_pct": top_micro_stock.get("day_change_pct", 0.0),
                    "source": top_micro_stock.get("source", "Micro-Asset")
                })
                stocks_total = cost

        remaining_for_mf = round(amount - stocks_total, 2)
        mf_allocations = [{
            "name": primary_mf.get("name", "Core Index Growth Fund (₹100 SIP)"),
            "allocated_amount": remaining_for_mf,
            "weight_pct": round((remaining_for_mf / amount) * 100, 1),
            "category": primary_mf.get("category", "Student Index Fund"),
            "cagr_5y": base_cagr,
            "volatility": base_vol,
            "nav": primary_mf.get("nav", 100.0),
            "nav_date": primary_mf.get("nav_date", "-"),
            "day_change_pct": primary_mf.get("day_change_pct", 0.0),
            "source": primary_mf.get("source", "Index Mutual Fund")
        }]

        sip_monthly = 100.0 if currency_code == "INR" else 10.0
        if amount >= 300.0:
            sip_monthly = 250.0 if currency_code == "INR" else 20.0

        advice = (
            f"🎓 Student & Secondary Income Strategy ({currency} {amount:,.0f}): Even putting {currency} 100 to {currency} 500 "
            f"into micro-SIPs or fractional ETFs gives you immediate ownership in India's top corporate leaders. "
            f"Starting with {currency} {sip_monthly:.0f}/month during college years compounds into a substantial financial cushion upon graduation!"
        )

        sip_info = {
            "total_allocated": remaining_for_mf,
            "monthly_recommended": sip_monthly,
            "recommended_funds": [
                {"name": primary_mf.get("name", "Navi/ICICI Nifty 50 Index Fund"), "monthly_amount": sip_monthly, "frequency": "Monthly", "category": "₹100 Student SIP"}
            ],
            "advice": f"Set up an automated {currency} {sip_monthly:,.0f}/month student micro-SIP via UPI or standing auto-debit."
        }

        return {
            "strategy": "🎓 Student & Secondary Income Wealth Builder (₹100 - ₹500)",
            "special_note": advice,
            "is_micro": True,
            "currency": currency,
            "total_capital": amount,
            "stocks_total": stocks_total,
            "stocks_pct": round((stocks_total / amount) * 100, 1),
            "stocks_picks": stock_picks,
            "mfs_total": remaining_for_mf,
            "mfs_pct": round((remaining_for_mf / amount) * 100, 1),
            "mfs_picks": mf_allocations,
            "sip_total": remaining_for_mf,
            "sip_pct": round((remaining_for_mf / amount) * 100, 1),
            "sip_info": sip_info,
            "cash_buffer": 0.0,
            "expected_cagr": round(base_cagr, 2),
            "expected_volatility": round(base_vol, 2),
            "sharpe_ratio": round((base_cagr / 100.0 - rf_rate) / (base_vol / 100.0), 2),
            "mpt_optimal_weights": {"Direct Stocks": round(stocks_total/amount, 2), "Mutual Funds": round(remaining_for_mf/amount, 2)},
            "efficient_frontier": None
        }

    # ------------------ CASE 2: Dynamic MPT Optimization ------------------
    # Select top quality candidate stocks
    candidate_stocks = stocks_available[:6]
    n_assets = len(candidate_stocks)

    raw_rets = np.array([s.get("cagr_5y", 15.0) / 100.0 for s in candidate_stocks])
    # Apply sentiment momentum tilt to expected returns
    adjusted_rets = raw_rets + (sentiment_adjustment / 100.0)
    raw_vols = np.array([s.get("volatility", 18.0) / 100.0 for s in candidate_stocks])

    # Check for real-time empirical covariance matrix from live market data
    empirical_cov = universe.get("empirical_cov")
    candidate_symbols = [s["symbol"] for s in candidate_stocks]
    if empirical_cov is not None and all(sym in empirical_cov.columns for sym in candidate_symbols):
        cov_matrix = empirical_cov.loc[candidate_symbols, candidate_symbols].to_numpy()
    else:
        cov_matrix = build_synthetic_covariance(raw_vols, avg_correlation=0.32)

    # Solve MPT Optimization
    opt_weights, opt_ret, opt_vol, opt_sharpe = optimize_max_sharpe(adjusted_rets, cov_matrix, rf_rate)
    gmv_weights, gmv_ret, gmv_vol, gmv_sharpe = optimize_min_variance(adjusted_rets, cov_matrix, rf_rate)
    ef_data = generate_efficient_frontier(adjusted_rets, cov_matrix, rf_rate)

    # Macro asset class allocation breakdown
    profile = ALLOCATION_PROFILES.get(risk_profile, ALLOCATION_PROFILES["Moderate"])
    
    # If Conservative, lean towards GMV weights; if Aggressive/Moderate, lean towards Max Sharpe
    if risk_profile == "Conservative":
        chosen_stock_weights = gmv_weights
        stocks_budget = amount * (profile["stocks_pct"] / 100.0)
        mfs_budget = amount * (profile["mutual_funds_pct"] / 100.0)
        sip_budget = amount * (profile["sip_pct"] / 100.0)
    else:
        chosen_stock_weights = opt_weights
        stocks_budget = amount * (profile["stocks_pct"] / 100.0)
        mfs_budget = amount * (profile["mutual_funds_pct"] / 100.0)
        sip_budget = amount * (profile["sip_pct"] / 100.0)

    # Calculate discrete share allocations for direct equity
    stock_picks = []
    stock_spend = 0.0

    for i, stk in enumerate(candidate_stocks):
        target_allocation = stocks_budget * chosen_stock_weights[i]
        price = max(stk.get("price", 100.0), 1.0)
        qty = int(target_allocation // price)
        if qty < 1 and (stocks_budget - stock_spend) >= price:
            qty = 1  # Allocate at least 1 unit if affordable within remaining budget

        cost = qty * price
        if qty > 0 and (stock_spend + cost) <= stocks_budget:
            stock_picks.append({
                "symbol": stk.get("symbol"),
                "name": stk.get("name"),
                "price": price,
                "quantity": qty,
                "allocated_amount": round(cost, 2),
                "sector": stk.get("sector"),
                "cagr_5y": round(stk.get("cagr_5y", 15.0) + sentiment_adjustment, 1),
                "volatility": stk.get("volatility", 18.0),
                "pe_ratio": stk.get("pe_ratio", 25.0),
                "mpt_weight_pct": round(chosen_stock_weights[i] * 100.0, 1),
                "day_change_pct": stk.get("day_change_pct", 0.0),
                "high_52w": stk.get("high_52w", price * 1.15),
                "low_52w": stk.get("low_52w", price * 0.85),
                "source": stk.get("source", "Live Market (Real-Time)")
            })
            stock_spend += cost

    total_stocks_invested = sum(s["allocated_amount"] for s in stock_picks)
    for s in stock_picks:
        s["weight_pct"] = round((s["allocated_amount"] / max(total_stocks_invested, 1.0)) * 100, 1)

    # Residual stock change sweeps into Mutual Funds
    residual_stocks = stocks_budget - total_stocks_invested
    mfs_budget += max(residual_stocks, 0.0)

    # Mutual Funds allocation
    mf_picks = []
    if mfs_available:
        if len(mfs_available) >= 2:
            amt_mf1 = round(mfs_budget * 0.55, 2)
            amt_mf2 = round(mfs_budget * 0.45, 2)
            mf_picks.append({
                "name": mfs_available[0]["name"],
                "category": mfs_available[0]["category"],
                "allocated_amount": amt_mf1,
                "weight_pct": 55.0,
                "cagr_5y": round(mfs_available[0]["cagr_5y"] + sentiment_adjustment * 0.5, 1),
                "volatility": mfs_available[0]["volatility"],
                "nav": mfs_available[0].get("nav", 100.0),
                "nav_date": mfs_available[0].get("nav_date", ""),
                "day_change_pct": mfs_available[0].get("day_change_pct", 0.0),
                "source": mfs_available[0].get("source", "Live AMFI Feed")
            })
            mf_picks.append({
                "name": mfs_available[1]["name"],
                "category": mfs_available[1]["category"],
                "allocated_amount": amt_mf2,
                "weight_pct": 45.0,
                "cagr_5y": round(mfs_available[1]["cagr_5y"] + sentiment_adjustment * 0.5, 1),
                "volatility": mfs_available[1]["volatility"],
                "nav": mfs_available[1].get("nav", 100.0),
                "nav_date": mfs_available[1].get("nav_date", ""),
                "day_change_pct": mfs_available[1].get("day_change_pct", 0.0),
                "source": mfs_available[1].get("source", "Live AMFI Feed")
            })
        else:
            mf_picks.append({
                "name": mfs_available[0]["name"],
                "category": mfs_available[0]["category"],
                "allocated_amount": mfs_budget,
                "weight_pct": 100.0,
                "cagr_5y": round(mfs_available[0]["cagr_5y"] + sentiment_adjustment * 0.5, 1),
                "volatility": mfs_available[0]["volatility"],
                "nav": mfs_available[0].get("nav", 100.0),
                "nav_date": mfs_available[0].get("nav_date", ""),
                "day_change_pct": mfs_available[0].get("day_change_pct", 0.0),
                "source": mfs_available[0].get("source", "Live AMFI Feed")
            })
    total_mfs_invested = sum(m["allocated_amount"] for m in mf_picks)

    # Monthly Systematic Investment Plan (SIP) schedule
    monthly_sip = round(sip_budget / 6.0, 2) if sip_budget > 0 else 500.0
    sip_picks = []
    if mfs_available:
        sip_picks.append({
            "name": mfs_available[0]["name"],
            "monthly_amount": round(monthly_sip * 0.60, 2),
            "frequency": "Monthly",
            "category": "Core Index SIP"
        })
        if len(mfs_available) > 1:
            sip_picks.append({
                "name": mfs_available[1]["name"],
                "monthly_amount": round(monthly_sip * 0.40, 2),
                "frequency": "Monthly",
                "category": "Growth Flexi-Cap SIP"
            })

    sip_info = {
        "total_allocated": round(sip_budget, 2),
        "monthly_recommended": monthly_sip,
        "recommended_funds": sip_picks,
        "advice": (
            f"Reserve {currency} {sip_budget:,.2f} to deploy as systematic monthly DCA of "
            f"{currency} {monthly_sip:,.2f}/month over the next 6 months to minimize timing volatility."
        )
    }

    # Blended Portfolio Metrics
    weights_and_stats = []
    for s in stock_picks:
        weights_and_stats.append((s["allocated_amount"] / amount, s["cagr_5y"], s["volatility"]))
    for m in mf_picks:
        weights_and_stats.append((m["allocated_amount"] / amount, m["cagr_5y"], m["volatility"]))
    if sip_picks:
        avg_mf_cagr = np.mean([m["cagr_5y"] for m in mf_picks]) if mf_picks else 15.0
        avg_mf_vol = np.mean([m["volatility"] for m in mf_picks]) if mf_picks else 14.0
        weights_and_stats.append((sip_budget / amount, avg_mf_cagr, avg_mf_vol))

    blended_cagr = sum(w * r for w, r, v in weights_and_stats)
    blended_vol = sum(w * v for w, r, v in weights_and_stats)
    blended_sharpe = (blended_cagr / 100.0 - rf_rate) / (blended_vol / 100.0) if blended_vol > 0 else 0.0

    total_deployed = total_stocks_invested + total_mfs_invested + sip_budget
    cash_buffer = max(round(amount - total_deployed, 2), 0.0)

    return {
        "strategy": f"{risk_profile} Markowitz MPT Optimized",
        "special_note": f"Modern Portfolio Theory allocation maximizing risk-adjusted return (Sharpe Ratio: {blended_sharpe:.2f}).",
        "is_micro": False,
        "currency": currency,
        "total_capital": amount,
        "stocks_total": round(total_stocks_invested, 2),
        "stocks_pct": round((total_stocks_invested / amount) * 100, 1),
        "stocks_picks": stock_picks,
        "mfs_total": round(total_mfs_invested, 2),
        "mfs_pct": round((total_mfs_invested / amount) * 100, 1),
        "mfs_picks": mf_picks,
        "sip_total": round(sip_budget, 2),
        "sip_pct": round((sip_budget / amount) * 100, 1),
        "sip_info": sip_info,
        "cash_buffer": cash_buffer,
        "expected_cagr": round(blended_cagr, 2),
        "expected_volatility": round(blended_vol, 2),
        "sharpe_ratio": round(blended_sharpe, 2),
        "optimal_point": {"volatility": round(opt_vol * 100, 2), "return": round(opt_ret * 100, 2), "sharpe": round(opt_sharpe, 2)},
        "gmv_point": {"volatility": round(gmv_vol * 100, 2), "return": round(gmv_ret * 100, 2), "sharpe": round(gmv_sharpe, 2)},
        "efficient_frontier": ef_data
    }
