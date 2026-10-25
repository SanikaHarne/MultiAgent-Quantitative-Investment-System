"""
Agent G: Quantitative Strategy Lab & Backtesting Engine
Implements historical simulation for algorithmic trading rules:
1. EMA Trend-Following (EMA 20 / EMA 50 Crossover)
2. RSI Mean Reversion (RSI < 30 Oversold Entry, RSI > 65 Exit)
3. Momentum High Breakout (20-day Donchian High Breakout)
4. Bollinger Band Mean Reversion
Computes Total Return, CAGR, Max Drawdown, Sharpe Ratio, Win Rate, Profit Factor,
and applies realistic transaction costs and slippage (0.10% per trade).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class QuantitativeStrategyAgent:
    """
    Backtests quantitative strategies on historical data with realistic
    friction assumptions, equity curve modeling, and risk-adjusted metrics.
    """

    def __init__(self, risk_free_rate: float = 0.05, transaction_cost_pct: float = 0.10):
        self.risk_free_rate = risk_free_rate
        self.transaction_cost_pct = transaction_cost_pct / 100.0  # 0.001 per trade

    def backtest(
        self,
        df_ohlcv: pd.DataFrame,
        strategy_name: str = "EMA Trend Following",
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Executes backtest over the provided OHLCV history.
        """
        if df_ohlcv.empty or len(df_ohlcv) < 60:
            return {
                "status": "INSUFFICIENT_DATA",
                "summary": "At least 60 daily bars required to execute strategy backtest.",
                "metrics": {},
                "equity_curve": pd.DataFrame()
            }

        df = df_ohlcv.copy()
        close = df["Close"]

        # Generate Trading Signals (1 = Long, 0 = Flat/Cash)
        signals = pd.Series(0, index=df.index)

        if "EMA Trend" in strategy_name:
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            signals = (ema20 > ema50).astype(int)

        elif "RSI Mean Reversion" in strategy_name:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            # State machine: enter when RSI < 32, hold until RSI > 62
            pos = 0
            pos_series = []
            for r in rsi:
                if np.isnan(r):
                    pos_series.append(0)
                elif r < 32:
                    pos = 1
                    pos_series.append(1)
                elif r > 62:
                    pos = 0
                    pos_series.append(0)
                else:
                    pos_series.append(pos)
            signals = pd.Series(pos_series, index=df.index)

        elif "Breakout" in strategy_name:
            high_20 = df["High"].shift(1).rolling(window=20).max()
            low_10 = df["Low"].shift(1).rolling(window=10).min()
            pos = 0
            pos_series = []
            for c, h, l in zip(close, high_20, low_10):
                if np.isnan(h):
                    pos_series.append(0)
                elif c > h:
                    pos = 1
                    pos_series.append(1)
                elif c < l:
                    pos = 0
                    pos_series.append(0)
                else:
                    pos_series.append(pos)
            signals = pd.Series(pos_series, index=df.index)

        else:
            # Bollinger Band Mean Reversion
            mid = close.rolling(20).mean()
            std = close.rolling(20).std()
            lower = mid - (2.0 * std)
            upper = mid + (2.0 * std)
            pos = 0
            pos_series = []
            for c, l, u in zip(close, lower, upper):
                if np.isnan(l):
                    pos_series.append(0)
                elif c <= l:
                    pos = 1
                    pos_series.append(1)
                elif c >= u:
                    pos = 0
                    pos_series.append(0)
                else:
                    pos_series.append(pos)
            signals = pd.Series(pos_series, index=df.index)

        # Lag signals by 1 bar to prevent lookahead bias (trade executes on NEXT day's open/close)
        trade_positions = signals.shift(1).fillna(0)

        # Daily raw stock return
        stock_returns = close.pct_change().fillna(0)

        # Strategy turnover (trades triggered)
        trades_executed = trade_positions.diff().abs().fillna(0)
        cost_drag = trades_executed * self.transaction_cost_pct

        # Strategy net daily returns
        strat_returns = (trade_positions * stock_returns) - cost_drag

        # Cumulative Equity Curves
        cum_strat = (1.0 + strat_returns).cumprod() * initial_capital
        cum_benchmark = (1.0 + stock_returns).cumprod() * initial_capital

        # Metrics calculation
        n_days = len(strat_returns)
        n_years = max(n_days / 252.0, 0.25)

        total_return_pct = ((cum_strat.iloc[-1] / initial_capital) - 1.0) * 100.0
        bench_return_pct = ((cum_benchmark.iloc[-1] / initial_capital) - 1.0) * 100.0

        cagr_pct = ((cum_strat.iloc[-1] / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0
        bench_cagr_pct = ((cum_benchmark.iloc[-1] / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0

        # Max Drawdown
        rolling_peak = cum_strat.cummax()
        drawdown_series = (cum_strat - rolling_peak) / rolling_peak
        max_drawdown_pct = float(drawdown_series.min() * 100.0)

        bench_peak = cum_benchmark.cummax()
        bench_dd = (cum_benchmark - bench_peak) / bench_peak
        bench_max_dd_pct = float(bench_dd.min() * 100.0)

        # Volatility & Sharpe Ratio
        strat_vol_ann = float(strat_returns.std() * np.sqrt(252.0) * 100.0)
        excess_return = (cagr_pct / 100.0) - self.risk_free_rate
        sharpe_ratio = round(excess_return / (strat_vol_ann / 100.0), 2) if strat_vol_ann > 0 else 0.0

        # Trade Count & Win Rate (grouping trade intervals)
        in_trade = False
        trade_pnls = []
        entry_equity = initial_capital

        for pos_val, eq in zip(trade_positions, cum_strat):
            if pos_val == 1 and not in_trade:
                in_trade = True
                entry_equity = eq
            elif pos_val == 0 and in_trade:
                in_trade = False
                trade_pnls.append(eq - entry_equity)

        total_trades = len(trade_pnls)
        if total_trades > 0:
            winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
            win_rate_pct = round((winning_trades / total_trades) * 100.0, 1)
            gross_profit = sum(p for p in trade_pnls if p > 0)
            gross_loss = abs(sum(p for p in trade_pnls if p < 0))
            profit_factor = round(gross_profit / (gross_loss + 1e-9), 2)
        else:
            win_rate_pct = 0.0
            profit_factor = 0.0

        # Construct equity DataFrame for chart plotting
        equity_df = pd.DataFrame({
            "Date": df.index,
            "Strategy_Equity": cum_strat.values,
            "Benchmark_Equity": cum_benchmark.values,
            "Drawdown": (drawdown_series.values * 100.0)
        }).set_index("Date")

        summary = (
            f"Strategy '{strategy_name}' generated a Total Return of {total_return_pct:+.1f}% "
            f"(CAGR: {cagr_pct:.1f}% vs Benchmark: {bench_cagr_pct:.1f}%) over {n_years:.1f} years. "
            f"Max Drawdown: {max_drawdown_pct:.1f}% (Benchmark: {bench_max_dd_pct:.1f}%). "
            f"Executed {total_trades} round-trip trades with a {win_rate_pct}% Win Rate and Sharpe Ratio of {sharpe_ratio:.2f}."
        )

        return {
            "status": "SUCCESS",
            "strategy_name": strategy_name,
            "initial_capital": initial_capital,
            "final_equity": round(float(cum_strat.iloc[-1]), 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr_pct, 2),
            "benchmark_return_pct": round(bench_return_pct, 2),
            "benchmark_cagr_pct": round(bench_cagr_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "benchmark_max_drawdown_pct": round(bench_max_dd_pct, 2),
            "annualized_volatility_pct": round(strat_vol_ann, 1),
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "win_rate_pct": win_rate_pct,
            "profit_factor": profit_factor,
            "transaction_cost_applied_pct": round(self.transaction_cost_pct * 100.0, 2),
            "equity_df": equity_df,
            "summary": summary
        }
