"""
Visualization Module:
Generates interactive, publication-quality Plotly financial charts
for asset allocation, scenario projections, Markowitz Efficient Frontier,
sentiment gauges, technical candlestick charts, and options payoffs.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

THEME_COLORS = {
    "stocks": "#3B82F6",       # Blue
    "mutual_funds": "#10B981", # Emerald Green
    "sip": "#F59E0B",          # Amber/Orange
    "best": "#10B981",         # Green
    "avg": "#06B6D4",          # Cyan
    "worst": "#EF4444",        # Red/Coral
    "capital": "#94A3B8",      # Slate/Gray
    "gold": "#FBBF24"          # Gold/Yellow
}


def plot_asset_allocation(portfolio: Dict[str, Any], currency: str) -> go.Figure:
    """Renders an interactive Donut chart showing asset allocation across Stocks, Mutual Funds, and SIP."""
    labels = ["Direct Stocks", "Mutual Funds (Lump Sum)", "Systematic Investment Plan (SIP)"]
    values = [portfolio.get("stocks_total", 0.0), portfolio.get("mfs_total", 0.0), portfolio.get("sip_total", 0.0)]
    colors = [THEME_COLORS["stocks"], THEME_COLORS["mutual_funds"], THEME_COLORS["sip"]]

    filtered_labels, filtered_values, filtered_colors = [], [], []
    for l, v, c in zip(labels, values, colors):
        if v > 0:
            filtered_labels.append(l)
            filtered_values.append(v)
            filtered_colors.append(c)

    fig = go.Figure(data=[go.Pie(
        labels=filtered_labels,
        values=filtered_values,
        hole=0.55,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Amount: " + currency + " %{value:,.2f}<br>Share: %{percent}<extra></extra>",
        marker=dict(colors=filtered_colors, line=dict(color="#1E293B", width=2))
    )])

    fig.update_layout(
        title=dict(text="<b>Recommended Capital Allocation</b>", font=dict(size=18, color="#F8FAFC")),
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#94A3B8")),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_scenario_projections(sim_result: Dict[str, Any], currency: str) -> go.Figure:
    """Renders the 98% Statistical Confidence Cone (Worst, Average, Best Case)."""
    df = sim_result["df_projections"]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["Year"], y=df["Best_Case"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Worst_Case"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(16, 185, 129, 0.12)", showlegend=False))

    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Best_Case"], mode="lines+markers", name="🔥 Best Case Scenario (Top 2%)",
        line=dict(color=THEME_COLORS["best"], width=2.5, dash="dot"), marker=dict(size=6),
        hovertemplate="<b>Best Case (Year %{x})</b>: " + currency + " %{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Average_Case"], mode="lines+markers", name="🎯 Average Expected Case (Median)",
        line=dict(color=THEME_COLORS["avg"], width=3.5), marker=dict(size=7),
        hovertemplate="<b>Average Case (Year %{x})</b>: " + currency + " %{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Worst_Case"], mode="lines+markers", name="⚠️ Worst Case Scenario (Stress-Tested)",
        line=dict(color=THEME_COLORS["worst"], width=2.5, dash="dash"), marker=dict(size=6),
        hovertemplate="<b>Worst Case (Year %{x})</b>: " + currency + " %{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Invested_Capital"], mode="lines", name="💼 Total Invested Capital",
        line=dict(color=THEME_COLORS["capital"], width=2, dash="dashdot"),
        hovertemplate="<b>Invested Capital</b>: " + currency + " %{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Stochastic Wealth Growth Projections (98% Confidence Cone)</b>", font=dict(size=18, color="#F8FAFC")),
        xaxis=dict(title="Investment Horizon (Years)", dtick=1, gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        yaxis=dict(title=f"Projected Portfolio Value ({currency})", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, font=dict(color="#94A3B8")),
        hovermode="x unified",
        height=480,
        margin=dict(t=60, b=50, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_efficient_frontier(ef_data: Dict[str, Any], optimal_point: Dict[str, Any], gmv_point: Dict[str, Any]) -> go.Figure:
    """Renders Markowitz Efficient Frontier curve with simulated portfolio cloud."""
    fig = go.Figure()

    if "random_vols" in ef_data and len(ef_data["random_vols"]) > 0:
        fig.add_trace(go.Scatter(
            x=ef_data["random_vols"], y=ef_data["random_rets"], mode="markers",
            marker=dict(size=5, color=ef_data["random_sharpes"], colorscale="Viridis", colorbar=dict(title=dict(text="Sharpe Ratio", font=dict(color="#94A3B8")), tickfont=dict(color="#94A3B8")), opacity=0.65),
            name="Feasible Portfolios"
        ))

    if "frontier_vol" in ef_data and len(ef_data["frontier_vol"]) > 0:
        fig.add_trace(go.Scatter(
            x=ef_data["frontier_vol"], y=ef_data["frontier_ret"], mode="lines",
            name="Markowitz Efficient Frontier", line=dict(color="#38BDF8", width=3.5)
        ))

    fig.add_trace(go.Scatter(
        x=[optimal_point["volatility"]], y=[optimal_point["return"]], mode="markers+text",
        name="⭐ Max Sharpe Portfolio (Tangency)", text=["Max Sharpe"], textposition="top right",
        textfont=dict(color=THEME_COLORS["gold"], size=12),
        marker=dict(symbol="star", size=17, color=THEME_COLORS["gold"], line=dict(color="#000", width=1.5))
    ))

    fig.add_trace(go.Scatter(
        x=[gmv_point["volatility"]], y=[gmv_point["return"]], mode="markers+text",
        name="🔷 Global Min Variance (GMV)", text=["GMV"], textposition="bottom left",
        textfont=dict(color="#A78BFA", size=12),
        marker=dict(symbol="diamond", size=14, color="#A78BFA", line=dict(color="#000", width=1.5))
    ))

    fig.update_layout(
        title=dict(text="<b>Markowitz Modern Portfolio Theory (MPT) Efficient Frontier</b>", font=dict(size=18, color="#F8FAFC")),
        xaxis=dict(title="Annualized Volatility / Risk (σ %)", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        yaxis=dict(title="Expected Annual Return (μ %)", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, font=dict(color="#94A3B8")),
        height=480,
        margin=dict(t=60, b=50, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_sentiment_gauge(score: float, label: str) -> go.Figure:
    """Renders an indicator gauge for composite market news sentiment (-1.0 to +1.0)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(text=f"<b>Composite Market Sentiment: {label}</b>", font=dict(size=16, color="#F8FAFC")),
        delta=dict(reference=0.0, increasing=dict(color="#10B981"), decreasing=dict(color="#EF4444")),
        gauge=dict(
            axis=dict(range=[-1.0, 1.0], tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0], ticktext=["Bearish", "-0.5", "Neutral", "+0.5", "Bullish"], tickcolor="#94A3B8"),
            bar=dict(color="#38BDF8", thickness=0.3),
            bgcolor="#1E293B",
            steps=[
                dict(range=[-1.0, -0.15], color="rgba(239, 68, 68, 0.35)"),
                dict(range=[-0.15, 0.15], color="rgba(148, 163, 184, 0.25)"),
                dict(range=[0.15, 1.0], color="rgba(16, 185, 129, 0.35)")
            ]
        )
    ))

    fig.update_layout(
        height=260,
        margin=dict(t=50, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC")
    )
    return fig


def plot_holdings_breakup(portfolio: Dict[str, Any], currency: str) -> go.Figure:
    """Renders a bar chart showing the capital allocated to each specific asset."""
    items = []
    for s in portfolio.get("stocks_picks", []):
        items.append({"Asset": f"{s['name']} ({s['symbol']})", "Type": "Stock", "Amount": s["allocated_amount"], "CAGR": s.get("cagr_5y", 15.0)})
    for m in portfolio.get("mfs_picks", []):
        items.append({"Asset": m["name"], "Type": "Mutual Fund", "Amount": m["allocated_amount"], "CAGR": m.get("cagr_5y", 15.0)})
    if portfolio.get("sip_total", 0) > 0:
        items.append({"Asset": "Systematic Monthly SIP Reserve", "Type": "SIP", "Amount": portfolio["sip_total"], "CAGR": portfolio.get("expected_cagr", 14.0)})

    if not items:
        return go.Figure()

    df = pd.DataFrame(items).sort_values(by="Amount", ascending=True)
    color_map = {"Stock": THEME_COLORS["stocks"], "Mutual Fund": THEME_COLORS["mutual_funds"], "SIP": THEME_COLORS["sip"]}

    fig = px.bar(df, x="Amount", y="Asset", color="Type", orientation="h", color_discrete_map=color_map, title="<b>Capital Deployed Per Selected Asset</b>")

    fig.update_layout(
        xaxis=dict(title=f"Amount ({currency})", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        yaxis=dict(title="", color="#94A3B8"),
        height=380,
        margin=dict(t=50, b=30, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="#94A3B8")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_wealth_comparison_bars(sim_result: Dict[str, Any], currency: str) -> go.Figure:
    """Renders side-by-side comparison bars for Capital Invested vs Final Wealth."""
    categories = ["Invested Capital", "Worst Case (Bear)", "Average Case (Expected)", "Best Case (Bull)"]
    values = [sim_result["total_invested"], sim_result["final_worst"], sim_result["final_avg"], sim_result["final_best"]]
    colors = [THEME_COLORS["capital"], THEME_COLORS["worst"], THEME_COLORS["avg"], THEME_COLORS["best"]]

    fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors, text=[f"{currency} {v:,.0f}" for v in values], textposition="outside")])

    fig.update_layout(
        title=dict(text="<b>Final Wealth Comparison at Horizon</b>", font=dict(size=18, color="#F8FAFC")),
        yaxis=dict(title=f"Value ({currency})", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        xaxis=dict(color="#94A3B8"),
        height=380,
        margin=dict(t=50, b=40, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_candlestick_trading(df_ohlc: pd.DataFrame, ticker: str, currency: str) -> go.Figure:
    """Renders an interactive OHLC Candlestick chart with Bollinger Bands and EMAs."""
    if df_ohlc.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_ohlc.index, open=df_ohlc["Open"], high=df_ohlc["High"], low=df_ohlc["Low"], close=df_ohlc["Close"], name="OHLC Price"))

    if "EMA_20" in df_ohlc.columns:
        fig.add_trace(go.Scatter(x=df_ohlc.index, y=df_ohlc["EMA_20"], mode="lines", name="EMA 20", line=dict(color="#0EA5E9", width=1.5)))
    if "EMA_50" in df_ohlc.columns:
        fig.add_trace(go.Scatter(x=df_ohlc.index, y=df_ohlc["EMA_50"], mode="lines", name="EMA 50", line=dict(color="#F59E0B", width=1.5)))

    if "BB_Upper" in df_ohlc.columns and "BB_Lower" in df_ohlc.columns:
        fig.add_trace(go.Scatter(x=df_ohlc.index, y=df_ohlc["BB_Upper"], mode="lines", name="Upper Band", line=dict(color="rgba(100, 116, 139, 0.4)", dash="dot")))
        fig.add_trace(go.Scatter(x=df_ohlc.index, y=df_ohlc["BB_Lower"], mode="lines", name="Lower Band", line=dict(color="rgba(100, 116, 139, 0.4)", dash="dot")))

    fig.update_layout(
        title=f"<b>{ticker} Technical Chart & Bollinger Bands</b>",
        xaxis_rangeslider_visible=False,
        font=dict(color="#F8FAFC"),
        yaxis=dict(title=f"Price ({currency})", gridcolor="rgba(148, 163, 184, 0.15)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
        height=420,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_options_payoff(payoff_df: pd.DataFrame, currency: str) -> go.Figure:
    """Renders Call vs Put options payoff curves across underlying price movements."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=payoff_df["Simulated Price"], y=payoff_df["Call PnL"], mode="lines+markers", name="Call Option PnL", line=dict(color="#10B981", width=2.5)))
    fig.add_trace(go.Scatter(x=payoff_df["Simulated Price"], y=payoff_df["Put PnL"], mode="lines+markers", name="Put Option PnL", line=dict(color="#EF4444", width=2.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")

    fig.update_layout(
        title="<b>Options Expiration Payoff Matrix (Call vs. Put)</b>",
        font=dict(color="#F8FAFC"),
        xaxis=dict(title=f"Simulated Price ({currency})", gridcolor="rgba(148, 163, 184, 0.15)"),
        yaxis=dict(title=f"Profit / Loss ({currency})", gridcolor="rgba(148, 163, 184, 0.15)"),
        height=380,
        margin=dict(t=50, b=30, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_similarity_distribution(horizons_dict: Dict[str, Any], horizon_key: str = "20d") -> go.Figure:
    """Renders distribution of empirical subsequent returns across matched historical analogues."""
    h_data = horizons_dict.get(horizon_key, {})
    returns = h_data.get("returns_distribution", [])

    fig = go.Figure()
    if not returns:
        return fig

    # Histogram of outcomes
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=12,
        name="Historical Outcomes",
        marker=dict(color="rgba(56, 189, 248, 0.7)", line=dict(color="#38BDF8", width=1.5)),
        opacity=0.85
    ))

    # Median line
    med = h_data.get("median_return_pct", 0.0)
    fig.add_vline(x=med, line_dash="dash", line_color="#FBBF24", annotation_text=f"Median: {med:+.1f}%", annotation_font_color="#FBBF24", annotation_position="top right")

    # Zero line
    fig.add_vline(x=0.0, line_dash="dot", line_color="#94A3B8")

    fig.update_layout(
        title=f"<b>Empirical Outcome Distribution ({h_data.get('name', 'Horizon')})</b>",
        xaxis=dict(title="Subsequent Forward Return (%)", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        yaxis=dict(title="Analogue Frequency", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        font=dict(color="#F8FAFC"),
        height=340,
        margin=dict(t=50, b=30, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_strategy_equity_curve(equity_df: pd.DataFrame, strategy_name: str, currency_sym: str = "₹") -> go.Figure:
    """Renders Strategy Equity Curve vs Benchmark Buy & Hold and Drawdown."""
    if equity_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["Strategy_Equity"],
        mode="lines", name=f"Quant Strategy ({strategy_name})",
        line=dict(color="#10B981", width=2.8)
    ))

    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["Benchmark_Equity"],
        mode="lines", name="Benchmark (Buy & Hold)",
        line=dict(color="#64748B", width=1.8, dash="dash")
    ))

    fig.update_layout(
        title=f"<b>Strategy Lab: Historical Equity Curve ({strategy_name})</b>",
        xaxis=dict(title="Date", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        yaxis=dict(title=f"Portfolio Equity ({currency_sym})", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, font=dict(color="#94A3B8")),
        height=400,
        margin=dict(t=50, b=40, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_forecast_probabilities(horizons_dict: Dict[str, Any]) -> go.Figure:
    """Renders grouped bar chart of P(Gain) vs P(Loss) across forecast horizons."""
    labels = []
    prob_gains = []
    prob_losses = []

    for k in ["1d", "5d", "20d"]:
        if k in horizons_dict:
            item = horizons_dict[k]
            labels.append(f"{item.get('name', k)}")
            prob_gains.append(item.get("prob_positive_pct", 50.0))
            prob_losses.append(item.get("prob_negative_pct", 50.0))

    if not labels:
        return go.Figure()

    fig = go.Figure(data=[
        go.Bar(name="P(Positive Return)", x=labels, y=prob_gains, marker_color="#10B981", text=[f"{p:.1f}%" for p in prob_gains], textposition="auto"),
        go.Bar(name="P(Negative Return)", x=labels, y=prob_losses, marker_color="#EF4444", text=[f"{p:.1f}%" for p in prob_losses], textposition="auto")
    ])

    fig.add_hline(y=50, line_dash="dash", line_color="#94A3B8")

    fig.update_layout(
        barmode="group",
        title="<b>Calibrated Machine Learning Outcome Probabilities</b>",
        yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        xaxis=dict(color="#94A3B8"),
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="#94A3B8")),
        height=320,
        margin=dict(t=50, b=30, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_multi_agent_consensus(agent_votes: Dict[str, Any]) -> go.Figure:
    """Renders multi-agent stance breakdown."""
    agents = list(agent_votes.keys())
    scores = [agent_votes[a]["score"] for a in agents]
    colors = ["#10B981" if s > 0 else ("#EF4444" if s < 0 else "#94A3B8") for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=agents,
        orientation="h",
        marker_color=colors,
        text=[agent_votes[a]["stance"] for a in agents],
        textposition="auto"
    ))

    fig.update_layout(
        title="<b>Individual Agent Alignment & Stances</b>",
        xaxis=dict(title="Bearish (-1) ◄——— Neutral (0) ———► Bullish (+1)", range=[-1.2, 1.2], tickvals=[-1, 0, 1], ticktext=["Bearish", "Neutral", "Bullish"], gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
        yaxis=dict(color="#94A3B8"),
        font=dict(color="#F8FAFC"),
        height=280,
        margin=dict(t=50, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig