# 📈 Smart Analysis & Investment System
### Final-Year Major Project: A Multi-Asset Quantitative Portfolio Optimization, Real-Time Sentiment Analysis, and Stochastic Wealth Scenario Engine

---

## 🏛️ System Architecture Overview

The system combines classical financial econometrics with modern NLP sentiment analysis and stochastic simulation:

```
smart_investment_system/
├── app.py                     # Streamlit glassmorphic dashboard with tabs & action controls
├── portfolio_optimizer.py     # Markowitz Modern Portfolio Theory (MPT) & Sharpe Ratio maximization
├── sentiment_engine.py        # Real-time Google News RSS scraper & NLTK VADER sentiment analyzer
├── simulation_engine.py       # 98% Statistical confidence Monte Carlo GBM simulation engine
├── report_generator.py        # Institutional PDF executive memo generator (ReportLab)
├── database.py                # SQLite persistence via SQLAlchemy ORM (portfolios, simulations)
├── kaggle_data_loader.py      # Kaggle dataset parser (pre-packaged & custom uploaded CSVs)
├── visualization.py           # Interactive Plotly charts (Efficient Frontier, Sentiment Gauge, Cones)
├── data_fetcher.py            # Live market quotes (yfinance + Indian MFAPI) with offline fallbacks
├── config.py                  # Assets catalog, risk profiles, and benchmark parameters
├── data/                      # Pre-packaged Kaggle benchmark datasets (Nifty 50, S&P 500, Mutual Funds)
├── requirements.txt           # Production dependencies
├── run.bat                    # Windows 1-click launcher
└── README.md                  # Project documentation
```

---

## 🔬 Core Quantitative & Architectural Modules

### 1. Markowitz Modern Portfolio Theory (MPT) Optimization (`portfolio_optimizer.py`)
- **Optimization Formulation**: Solves constrained non-linear optimization using `scipy.optimize.minimize` (SLSQP):
  $$\max_{\mathbf{w}} \text{Sharpe}(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}} \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le 1$$
- **Tangency & GMV Portfolios**: Solves for both the Maximum Sharpe Ratio Portfolio (Tangency) and Global Minimum Variance (GMV) Portfolio.
- **Markowitz Efficient Frontier**: Evaluates the hyperbola of minimum variance portfolios for target expected returns across $[\min(\mu), \max(\mu)]$, with a 1,200 random portfolio Monte Carlo cloud.
- **Micro-Investment Routing**: Automatically detects sub-threshold investments (< ₹2,000 / $75) and routes 100% into fractional index funds and micro-SIPs.

### 2. Live Financial News Sentiment Analysis (`sentiment_engine.py`)
- **Real-Time News Ingestion**: Scrapes Google News RSS feeds for stock tickers and enterprise names.
- **Financial Lexicon VADER (`nltk.sentiment.vader`)**: Augmented with quantitative domain terms (*bullish, bearish, surge, dividend, outperformance, downgrade*).
- **Sentiment Momentum Tilt**: Calculates a weighted composite score $S \in [-1.0, +1.0]$ and adjusts expected return drift:
  $$\Delta \mu_{\text{sentiment}} = S_{\text{composite}} \times 1.5\%$$

### 3. Stochastic Wealth Scenario Engine (`simulation_engine.py`)
- **Geometric Brownian Motion (GBM)**: Simulates 5,000+ stochastic return paths for lump sum and recurring monthly SIP additions:
  $$S_t = S_0 \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma \sqrt{t} Z\right)$$
- **98% Statistical Confidence Bounding**: Employs $Z = \pm 2.054$ (2.5-sigma equivalent) to produce mathematically bounded **Best Case (98th percentile)**, **Average Case (50th percentile)**, and **Worst Case (2nd percentile)** trajectories.
- **Risk Metrics**: Evaluates 1-Year Value-at-Risk ($\text{VaR}_{98\%}$) and Maximum Historical Drawdown.

### 4. Automated PDF Report Generation & Database Persistence (`report_generator.py` & `database.py`)
- **ReportLab PDF Generator**: One-click download of a 2-page institutional executive summary memo with KPI tables, sentiment summary, MPT asset breakdown, and milestone projections.
- **SQLite & SQLAlchemy**: Persists user portfolios, simulation runs, timestamps, and client names for historical tracking.

### 5. Kaggle Dataset Integration & Universal Parser (`kaggle_data_loader.py`)
- **Pre-packaged Kaggle Benchmark Datasets**: Includes pre-loaded datasets in `data/`:
  - `kaggle_nifty50_historical.csv`: 5-year daily NIFTY 50 equities data (1,240+ trading days).
  - `kaggle_sp500_historical.csv`: 5-year daily S&P 500 equities data (1,250+ trading days).
  - `kaggle_mutual_funds_india.csv`: Indian mutual funds historical performance, AUM, and volatility ratings.
- **Universal Custom Kaggle CSV Uploader**: Drag-and-drop any CSV downloaded from Kaggle. Automatically detects schema (wide time-series or long transactional `Date-Ticker-Close`), pivots into price matrices, and feeds real-time returns directly into Markowitz MPT and Monte Carlo simulations!

---

## 🚀 How to Run the Application

### Method 1: VS Code
1. Open the project folder in VS Code:
   ```powershell
   code "C:\Users\HP\OneDrive\Desktop\smart_investment_system"
   ```
2. Press **`F5`** or open the integrated terminal (`Ctrl + ~`) and type:
   ```powershell
   streamlit run app.py
   ```

### Method 2: One-Click Windows Batch Runner
Double-click [`run.bat`](file:///C:/Users/HP/OneDrive/Desktop/smart_investment_system/run.bat).
The dashboard will launch in your browser at:
👉 **`http://localhost:8501`**
