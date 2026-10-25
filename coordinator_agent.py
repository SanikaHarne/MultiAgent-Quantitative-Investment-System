"""
Agent H: Decision Coordinator Agent
Orchestrates the entire multi-agent quantitative pipeline (Agents A through G).
Compares agreement and conflict across Technicals, News Sentiment, Historical Similarity,
ML Forecasts, and Risk Controls.
Enforces abstention when evidence is contradictory or data is stale/insufficient.
Produces an explainable consolidated investment and trading thesis.
"""

from typing import Dict, Any, List, Optional
from market_data_agent import MarketDataAgent
from news_agent import NewsIntelligenceAgent
from technical_agent import TechnicalAnalysisAgent
from similarity_agent import HistoricalSimilarityAgent
from forecasting_agent import ForecastingAgent
from risk_agent import RiskManagementAgent
from strategy_agent import QuantitativeStrategyAgent


class MultiAgentCoordinator:
    """
    Central orchestration and consensus engine for QuantNexa.
    """

    def __init__(self, market: str = "IN", capital_budget: float = 25000.0):
        self.market = market
        self.capital_budget = capital_budget
        self.currency_sym = "₹" if market == "IN" else "$"

        self.data_agent = MarketDataAgent(market=market)
        self.news_agent = NewsIntelligenceAgent(market=market)
        self.tech_agent = TechnicalAnalysisAgent()
        self.sim_agent = HistoricalSimilarityAgent(top_k=15, min_gap_days=12)
        self.forecast_agent = ForecastingAgent(min_samples=120)
        self.risk_agent = RiskManagementAgent(max_risk_budget_pct=1.5)
        self.strategy_agent = QuantitativeStrategyAgent()

    def run_full_investigation(
        self,
        symbol: str,
        user_capital: Optional[float] = None,
        risk_profile: str = "Moderate"
    ) -> Dict[str, Any]:
        """
        Executes complete multi-agent investigation workflow for a target symbol.
        """
        capital = user_capital or self.capital_budget

        # 1. Market Data Retrieval (Agent A)
        quote = self.data_agent.get_quote(symbol)
        clean_symbol = quote.get("symbol", symbol)
        ohlcv_data = self.data_agent.fetch_ohlcv(clean_symbol, period="3y")
        df_ohlcv = ohlcv_data.get("df")

        if df_ohlcv is None or df_ohlcv.empty:
            return {
                "symbol": clean_symbol,
                "status": "DATA_FAILURE",
                "final_verdict": "ABSTAIN — INSUFFICIENT MARKET DATA",
                "confidence_score": 0,
                "memo": f"Unable to retrieve verified market history for {clean_symbol}. All agents halted.",
                "agents_data": {}
            }

        # Benchmark index series for relative strength
        benchmark_df = self.data_agent.get_benchmark_series(period="3y")
        self.tech_agent.benchmark_df = benchmark_df

        # 2. News Intelligence (Agent B)
        news_res = self.news_agent.harvest_news(clean_symbol, limit=8)

        # 3. Technical Analysis (Agent C)
        tech_res = self.tech_agent.analyze(df_ohlcv, clean_symbol)

        # 4. Historical Similarity & Forward Outcomes (Agent D)
        sim_res = self.sim_agent.analyze(df_ohlcv, benchmark_df)

        # 5. Probabilistic ML Forecasting (Agent E)
        forecast_res = self.forecast_agent.forecast_horizons(df_ohlcv, benchmark_df)

        # 6. Quantitative Risk Management (Agent F)
        current_price = quote.get("last_price", 100.0)
        atr_val = tech_res.get("indicators", {}).get("atr", current_price * 0.02)
        support_level = tech_res.get("indicators", {}).get("support_1")

        risk_res = self.risk_agent.evaluate_risk(
            df_ohlcv=df_ohlcv,
            current_price=current_price,
            atr_value=atr_val,
            total_capital=capital,
            currency_sym=self.currency_sym,
            risk_reward_ratio=2.0,
            support_price=support_level
        )

        # 7. Strategy Backtest Benchmark (Agent G)
        strat_res = self.strategy_agent.backtest(df_ohlcv, strategy_name="EMA Trend Following", initial_capital=capital)

        # ----------------- COORDINATION & RECONCILIATION -----------------
        agreements = []
        conflicts = []

        # Vector scores: +1 (Bullish), -1 (Bearish), 0 (Neutral)
        agent_votes = {}

        # Technical Vote
        tech_score = tech_res.get("composite_score", 0)
        if tech_score >= 2:
            agent_votes["Technical"] = {"stance": "BULLISH", "weight": 1.2, "score": 1}
        elif tech_score <= -2:
            agent_votes["Technical"] = {"stance": "BEARISH", "weight": 1.2, "score": -1}
        else:
            agent_votes["Technical"] = {"stance": "NEUTRAL", "weight": 1.0, "score": 0}

        # News Vote
        news_comp = news_res.get("composite_score", 0.0)
        if news_comp >= 0.15:
            agent_votes["News"] = {"stance": "BULLISH", "weight": 1.0, "score": 1}
        elif news_comp <= -0.15:
            agent_votes["News"] = {"stance": "BEARISH", "weight": 1.0, "score": -1}
        else:
            agent_votes["News"] = {"stance": "NEUTRAL", "weight": 0.8, "score": 0}

        # Historical Similarity Vote (20-day horizon win rate)
        sim_20d = sim_res.get("horizons", {}).get("20d", {})
        sim_win_rate = sim_20d.get("win_rate_pct", 50.0)
        if sim_win_rate >= 62.0:
            agent_votes["Historical Similarity"] = {"stance": "BULLISH", "weight": 1.1, "score": 1}
        elif sim_win_rate <= 38.0:
            agent_votes["Historical Similarity"] = {"stance": "BEARISH", "weight": 1.1, "score": -1}
        else:
            agent_votes["Historical Similarity"] = {"stance": "NEUTRAL", "weight": 0.9, "score": 0}

        # Machine Learning Forecast Vote (20-day P(Gain))
        fc_20d = forecast_res.get("horizons", {}).get("20d", {})
        p_gain = fc_20d.get("prob_positive_pct", 50.0)
        if p_gain >= 55.0:
            agent_votes["ML Forecasting"] = {"stance": "BULLISH", "weight": 1.1, "score": 1}
        elif p_gain <= 45.0:
            agent_votes["ML Forecasting"] = {"stance": "BEARISH", "weight": 1.1, "score": -1}
        else:
            agent_votes["ML Forecasting"] = {"stance": "NEUTRAL", "weight": 0.9, "score": 0}

        # Identify conflicts (e.g. Technicals Bullish but News Bearish)
        stances = [v["stance"] for v in agent_votes.values() if v["stance"] != "NEUTRAL"]
        has_direct_conflict = ("BULLISH" in stances and "BEARISH" in stances)

        if has_direct_conflict:
            bulls = [k for k, v in agent_votes.items() if v["stance"] == "BULLISH"]
            bears = [k for k, v in agent_votes.items() if v["stance"] == "BEARISH"]
            conflicts.append(f"Significant cross-agent divergence: {', '.join(bulls)} indicate upside whereas {', '.join(bears)} indicate downside pressure.")
        else:
            agreements.append(f"Broad directional alignment observed across active quantitative agents.")

        # Weighted Decision Aggregation
        total_w = sum(v["weight"] for v in agent_votes.values())
        weighted_sum = sum(v["score"] * v["weight"] for v in agent_votes.values())
        norm_score = weighted_sum / (total_w + 1e-9)

        # Risk Override: if Risk Level is ELEVATED and direct conflict exists, FORCE ABSTENTION
        if risk_res.get("risk_level") == "ELEVATED" and has_direct_conflict:
            verdict = "ABSTAIN — HIGH RISK CONFLICT"
            confidence = 35
            rationale = "Elevated volatility and conflicting agent signals trigger mandatory capital protection abstention."
        elif norm_score >= 0.50 and not has_direct_conflict:
            verdict = "FAVORABLE (ACCUMULATE ON DIP)"
            confidence = min(int(55 + (norm_score * 35)), 88)
            rationale = "Harmonious positive convergence across technicals, historical pattern analogues, and ML probabilities."
        elif norm_score <= -0.50 and not has_direct_conflict:
            verdict = "DEFENSIVE (REDUCE / HEDGE)"
            confidence = min(int(55 + (abs(norm_score) * 35)), 88)
            rationale = "Multiple quantitative indicators reflect elevated downside vulnerability."
        elif has_direct_conflict:
            verdict = "NEUTRAL (WAIT FOR BREAKOUT / RESOLUTION)"
            confidence = 45
            rationale = "Cross-currents between technical indicators and recent news flow suggest market indecision."
        else:
            verdict = "NEUTRAL / BALANCED"
            confidence = 50
            rationale = "Indicators remain near equilibrium with no decisive edge."

        # Consolidated Trade Memo
        memo = {
            "symbol": clean_symbol,
            "last_price": current_price,
            "currency": quote.get("currency", "INR"),
            "currency_sym": self.currency_sym,
            "final_verdict": verdict,
            "confidence_score": confidence,
            "rationale": rationale,
            "key_agreements": agreements,
            "key_conflicts": conflicts,
            "agent_votes": agent_votes,
            "actionable_levels": {
                "recommended_entry": current_price,
                "dynamic_stop_loss": risk_res["stop_loss_price"],
                "target_1": risk_res["target_price"],
                "target_2": risk_res["extended_target"],
                "max_risk_amount": risk_res["max_loss_amount"],
                "recommended_position_shares": risk_res["recommended_shares"],
                "position_cost": risk_res["position_value"]
            },
            "risk_status": risk_res["risk_level"],
            "data_freshness": quote.get("timestamp", "Recent"),
            "disclaimer": "⚠️ This assessment is an automated multi-agent quantitative synthesis for research purposes. It does not constitute financial advice or guarantee future performance."
        }

        return {
            "symbol": clean_symbol,
            "status": "SUCCESS",
            "verdict": verdict,
            "confidence_score": confidence,
            "memo": memo,
            "agents": {
                "market_data": quote,
                "news": news_res,
                "technical": tech_res,
                "similarity": sim_res,
                "forecasting": forecast_res,
                "risk": risk_res,
                "strategy": strat_res
            }
        }
