"""
QuantNexa - Smart Investment System
Final-Year Major Project: A Multi-Asset Quantitative Portfolio Optimization,
Real-Time Trading Signals, Overnight News Sentiment, and Options Risk Engine.
Locked in High-End Dark Theme.
"""

import sys
import os
import streamlit as st
from streamlit import runtime

if not runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())

import pandas as pd
import numpy as np
from datetime import datetime

# Core Domain Modules
from config import CURRENCY_SYMBOLS, ALLOCATION_PROFILES
from data_fetcher import get_market_universe
from portfolio_optimizer import optimize_portfolio
from simulation_engine import run_monte_carlo_simulation
from sentiment_engine import analyze_portfolio_sentiment
from report_generator import generate_executive_pdf
import database
import kaggle_data_loader

# Trading & Risk Engines
from indicator_engine import analyze_technical_indicators
from risk_engine import calculate_trade_risk
from news_engine import harvest_overnight_news, reconcile_signal_with_news
from options_engine import simulate_options_payoff

# Multi-Agent Architecture
from coordinator_agent import MultiAgentCoordinator
from market_data_agent import MarketDataAgent, get_catalog, search_catalog, normalize_symbol
from strategy_agent import QuantitativeStrategyAgent

from visualization import (
    plot_asset_allocation,
    plot_scenario_projections,
    plot_holdings_breakup,
    plot_wealth_comparison_bars,
    plot_efficient_frontier,
    plot_sentiment_gauge,
    plot_candlestick_trading,
    plot_options_payoff,
    plot_similarity_distribution,
    plot_strategy_equity_curve,
    plot_forecast_probabilities,
    plot_multi_agent_consensus
)

# 1. Page Configuration
st.set_page_config(
    page_title="QuantNexa | Smart Investment System",
    page_icon="Logo.png" if os.path.exists("Logo.png") else "⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()

# QuantNexa visual system: spacious, calm, premium fintech styling

theme_css = """
<style>
    :root { --qn-bg:#080D18; --qn-panel:#101827; --qn-border:rgba(148,163,184,.16); --qn-muted:#9AA9BD; --qn-teal:#39E6C1; --qn-blue:#269DFF; }
    .stApp { background: radial-gradient(ellipse at 12% 0%, #14263A 0%, #0A1220 38%, #080D18 78%); color:#F4F7FB; }
    .block-container { max-width: 1440px; padding-top: 2.4rem; padding-bottom: 3.5rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0D1727,#0A101C); border-right:1px solid var(--qn-border); }
    [data-testid="stSidebar"] .block-container { padding-top:1.4rem; }
    h1,h2,h3 { letter-spacing:-.035em; }
    h2 { margin-top:1.4rem !important; }
    .hero-title { font-size:clamp(2.2rem,4vw,3.3rem); font-weight:850; letter-spacing:-.055em; line-height:1.04; background:linear-gradient(100deg,#FFFFFF 10%,#8DEFD9 58%,#269DFF 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; }
    .hero-sub { font-size:.98rem; color:var(--qn-muted); margin-top:.65rem; line-height:1.65; font-weight:400; }
    .glass-card { background:linear-gradient(145deg,rgba(20,32,49,.88),rgba(12,20,33,.86)); border:1px solid var(--qn-border); border-radius:22px; padding:28px; box-shadow:0 18px 45px rgba(0,0,0,.18); margin:18px 0 24px; }
    .metric-box { background:linear-gradient(145deg,rgba(21,34,52,.92),rgba(13,22,36,.92)); border:1px solid var(--qn-border); border-radius:18px; padding:22px 22px 20px; min-height:112px; box-shadow:0 10px 24px rgba(0,0,0,.12); }
    .metric-title { font-size:.76rem; font-weight:650; color:#9AABC0; text-transform:uppercase; letter-spacing:.09em; }
    .metric-num { font-size:clamp(1.25rem,2vw,1.8rem); font-weight:780; color:#F7FAFF; margin-top:10px; line-height:1.2; overflow-wrap:anywhere; }
    .metric-caption { font-size:.82rem; color:#91A3B9; margin-top:8px; font-weight:450; }
    .neon-pill { display:inline-flex; align-items:center; gap:7px; padding:7px 13px; border-radius:999px; font-size:.78rem; font-weight:650; }
    .pill-purple { background:rgba(139,92,246,.12); color:#D4C5FF; border:1px solid rgba(167,139,250,.35); }
    .pill-green { background:rgba(16,185,129,.1); color:#80F0CE; border:1px solid rgba(52,211,153,.32); }
    .pill-amber { background:rgba(245,158,11,.1); color:#FCD58A; border:1px solid rgba(245,158,11,.32); }
    .pill-rose { background:rgba(244,63,94,.1); color:#FDA4AF; border:1px solid rgba(244,63,94,.32); }
    .student-banner { background:linear-gradient(120deg,rgba(38,157,255,.10),rgba(57,230,193,.08)); border:1px solid var(--qn-border); border-radius:18px; padding:20px 24px; margin-bottom:26px; }
    .auth-shell { max-width:1000px; margin:2rem auto 3rem; }
    .auth-panel { background:linear-gradient(145deg,rgba(18,30,47,.96),rgba(11,19,32,.96)); border:1px solid rgba(133,166,197,.20); border-radius:24px; padding:clamp(22px,4vw,42px); box-shadow:0 24px 70px rgba(0,0,0,.28); }
    .auth-eyebrow { color:#70E8D0; font-size:.75rem; text-transform:uppercase; letter-spacing:.18em; font-weight:750; margin-bottom:14px; }
    .auth-title { color:#F8FAFC; font-size:clamp(1.8rem,3vw,2.55rem); font-weight:800; letter-spacing:-.05em; line-height:1.12; margin:0 0 12px; }
    .auth-copy { color:#9AA9BD; line-height:1.7; font-size:.98rem; margin-bottom:26px; }
    .auth-feature { border-top:1px solid var(--qn-border); padding:18px 0; color:#D5E1EF; }
    .auth-feature strong { color:#F8FAFC; display:block; margin-bottom:5px; }
    .auth-feature span { color:#91A3B9; font-size:.9rem; line-height:1.5; }
    div[data-testid="stTabs"] button { padding:12px 16px; }
    div[data-testid="stTextInput"] label, div[data-testid="stSelectbox"] label { color:#C6D2E1; font-weight:550; }
    .stButton button { border-radius:12px; min-height:46px; font-weight:650; transition:all .18s ease; }
    .stButton button[kind="primary"] { background:linear-gradient(100deg,#28CDB2,#168BEA); border:0; color:#04111D; }
    .stButton button[kind="primary"]:hover { filter:brightness(1.08); box-shadow:0 8px 24px rgba(38,157,255,.2); }
    input,textarea,[data-baseweb="select"] > div { border-radius:10px !important; }
    div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] { margin-top:8px; }
    div[data-testid="stTabs"] { margin-top:1.5rem; }
    hr { border-color:rgba(148,163,184,.16) !important; }
    @media (max-width: 760px) { .block-container { padding:1.2rem 1rem 2rem; } .metric-box { padding:16px; } }

    [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 18% 25%, #0d2940 0%, #07111f 42%, #050914 100%); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1600px; padding-top: 2rem; padding-bottom: 2rem; }
    .qn-brand {font-size:3rem;font-weight:850;color:#f8fafc;letter-spacing:-2px;margin:0 0 2rem}.qn-brand span{background:linear-gradient(90deg,#36f0bd,#08a8ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .qn-quote-card{position:relative;border:1px solid rgba(41,207,229,.48);background:linear-gradient(120deg,rgba(7,23,41,.86),rgba(8,18,34,.42));border-radius:18px;padding:25px 30px 20px;margin:8px 0 30px;max-width:720px;box-shadow:0 12px 40px rgba(0,0,0,.12)}
    .qn-quote-mark,.qn-quote-close{color:#2ce7c5;font-size:2.7rem;font-weight:900;line-height:.8}.qn-quote-text{font-size:clamp(1.35rem,2vw,2rem);line-height:1.45;color:#edf5ff;padding-left:26px}.qn-quote-text span{color:#36e6c1;font-weight:750}.qn-quote-by{padding:10px 0 0 26px;color:#8ca9c8;font-size:.9rem}.qn-quote-close{position:absolute;right:20px;bottom:12px}
    .qn-eyebrow{color:#39e8c6;font-weight:850;font-size:.82rem;letter-spacing:4px;margin:8px 0 14px}.qn-headline{font-size:clamp(2.3rem,3.2vw,3.45rem);line-height:1.08;letter-spacing:-1.5px;color:#f8fafc;font-weight:850;margin:0 0 18px}.qn-headline span{color:#13b9ef}.qn-description{color:#a8c0dc;font-size:1.05rem;line-height:1.7;max-width:700px;margin-bottom:28px}
    .qn-features{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:26px}.qn-feature{display:flex;gap:12px;align-items:flex-start;color:#edf5ff}.qn-feature-icon{min-width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,#132e4c,#0b1b30);border:1px solid rgba(53,175,255,.18);color:#4bc8ff;font-size:1.7rem}.qn-feature b{font-size:.92rem}.qn-feature p{color:#9fb6d0;font-size:.83rem;line-height:1.5;margin:.35rem 0 0}
    .qn-workspace-card{display:flex;align-items:center;gap:18px;padding:22px 24px;border-radius:24px;border:1px solid rgba(73,166,222,.25);background:linear-gradient(130deg,rgba(17,38,62,.9),rgba(9,21,37,.75));margin:0 0 30px}.qn-workspace-icon{height:56px;min-width:56px;border-radius:18px;background:#123457;color:#2fb9ff;display:flex;align-items:center;justify-content:center;font-size:1.8rem}.qn-workspace-label{font-size:.78rem;letter-spacing:3px;font-weight:850;color:#16b9ff;margin-bottom:7px}.qn-workspace-copy{color:#a9c1dc;font-size:.95rem;line-height:1.55}
    .stTabs [data-baseweb="tab-list"]{gap:22px;border-bottom:1px solid rgba(148,163,184,.2)}.stTabs [data-baseweb="tab"]{padding:12px 4px;color:#b6c7dc}.stTabs [aria-selected="true"]{color:#35e4c2!important}.stTabs [data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#31e4bc,#089dff);height:3px}
    .qn-form-heading{font-size:2.35rem;font-weight:800;letter-spacing:-1px;color:#f7fbff;margin:30px 0 4px}.qn-form-subtitle{font-size:1rem;color:#9eb5d0;margin-bottom:26px}
    [data-testid="stTextInput"] label,[data-testid="stSelectbox"] label{color:#dbeafe;font-weight:600}.stTextInput input{background:#18263a!important;border:1px solid #263c55!important;border-radius:11px!important;color:#f8fafc!important;min-height:48px}.stTextInput input:focus{border-color:#20d8c2!important;box-shadow:0 0 0 1px #20d8c2!important}.stButton>button[kind="primary"]{border:0;border-radius:12px;min-height:50px;color:#041421;font-weight:800;background:linear-gradient(100deg,#27e0b8,#079df6);box-shadow:0 8px 24px rgba(0,176,219,.15)}.stButton>button[kind="primary"]:hover{filter:brightness(1.08);border:0;color:#041421}
    @media(max-width:900px){.qn-features{grid-template-columns:1fr}.qn-headline{font-size:2.4rem}.qn-quote-card{padding:20px}.qn-workspace-card{padding:16px}.qn-form-heading{font-size:2rem}}
    [data-testid="stImage"] img { border-radius: 16px; }

    /* QuantNexa post-login dashboard refresh */
    .qn-dashboard-hero { position:relative; overflow:hidden; padding:28px 32px; margin:10px 0 24px; border:1px solid rgba(47,220,205,.25); border-radius:24px; background: radial-gradient(ellipse at 90% 15%,rgba(0,153,255,.18),transparent 36%), linear-gradient(120deg,rgba(17,39,61,.96),rgba(8,17,31,.96)); box-shadow:0 18px 48px rgba(0,0,0,.20); }
    .qn-dashboard-hero:after { content:""; position:absolute; right:-55px; top:-105px; width:260px; height:260px; border-radius:50%; border:1px solid rgba(48,223,198,.14); box-shadow:0 0 0 24px rgba(48,223,198,.025),0 0 0 50px rgba(38,157,255,.025); pointer-events:none; }
    .qn-dashboard-kicker { color:#43E7C7; text-transform:uppercase; letter-spacing:.19em; font-size:.72rem; font-weight:800; margin-bottom:10px; }
    .qn-dashboard-title { font-size:clamp(1.8rem,3vw,2.65rem); line-height:1.1; letter-spacing:-.045em; font-weight:850; color:#F5FAFF; margin:0; }
    .qn-dashboard-title span { background:linear-gradient(90deg,#47E8C5,#24A7FF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .qn-dashboard-copy { color:#9FB5CF; margin-top:10px; line-height:1.65; max-width:760px; font-size:.96rem; }
    .qn-section-label { color:#45E4C4; font-size:.74rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin:10px 0 14px; }
    .metric-box { transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease; }
    .metric-box:hover { transform:translateY(-3px); border-color:rgba(55,222,198,.40); box-shadow:0 14px 32px rgba(0,0,0,.24); }
    .stTabs [data-baseweb="tab-list"] { gap:10px; padding:7px; border:1px solid rgba(148,163,184,.13); border-radius:15px; background:rgba(11,20,33,.75); }
    .stTabs [data-baseweb="tab"] { border-radius:10px; padding:11px 15px; }
    .stTabs [aria-selected="true"] { background:linear-gradient(110deg,rgba(39,224,184,.13),rgba(7,157,246,.13)); }
    [data-testid="stSidebar"] [data-testid="stImage"] img { border-radius:14px; }
    @media(max-width:760px) { .qn-dashboard-hero {padding:22px 20px}.qn-dashboard-copy {font-size:.9rem}.metric-num {font-size:1.25rem} }

</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

# 3. Session State Initialization for Authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# Use bull.png as the full-screen background on the sign-in / sign-up page.
# Embed the image as base64 so Streamlit can load it reliably from the project folder.
if not st.session_state["authenticated"] and os.path.exists("bull.png"):
    import base64
    with open("bull.png", "rb") as _bull_file:
        _bull_background_b64 = base64.b64encode(_bull_file.read()).decode("utf-8")
    st.markdown(f"""
    <style>
      [data-testid="stAppViewContainer"] {{
        background-image:
          linear-gradient(90deg, rgba(3, 10, 22, 0.42) 0%, rgba(3, 10, 22, 0.56) 48%, rgba(3, 10, 22, 0.90) 100%),
          linear-gradient(0deg, rgba(3, 8, 18, 0.70) 0%, rgba(3, 8, 18, 0.06) 48%, rgba(3, 8, 18, 0.28) 100%),
          url("data:image/png;base64,{_bull_background_b64}");
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        background-repeat: no-repeat;
      }}
      [data-testid="stMainBlockContainer"] {{
        background: transparent !important;
      }}
      .qn-workspace-card {{
        background: linear-gradient(135deg, rgba(7, 18, 34, .92), rgba(7, 18, 34, .78)) !important;
        backdrop-filter: blur(16px);
      }}
      .stTabs, [data-testid="stTextInput"], [data-testid="stSelectbox"] {{
        background: transparent;
      }}
      @media (max-width: 900px) {{
        [data-testid="stAppViewContainer"] {{ background-position: 58% center; background-attachment: scroll; }}
      }}
    </style>
    """, unsafe_allow_html=True)

# =========================================================================
# 4. AUTHENTICATION SCREENS (LOGIN & SIGNUP PORTAL)
# =========================================================================
if not st.session_state["authenticated"]:
    # Spacious, editorial-style login page inspired by the QuantNexa concept.
    left, right = st.columns([1.08, 1], gap="large", vertical_alignment="center")

    with left:
        if os.path.exists("Logo.png"):
            st.image("Logo.png", width=220)
        else:
            st.markdown("<div class='qn-brand'>Quant<span>Nexa</span></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="qn-quote-card">
            <div class="qn-quote-mark">“</div>
            <div class="qn-quote-text">Invest with insights today,<br>for a <span>brighter tomorrow.</span></div>
            <div class="qn-quote-by">— QuantNexa</div>
            <div class="qn-quote-close">”</div>
        </div>
        """, unsafe_allow_html=True)

        hero_text_col, hero_art_col = st.columns([1.18, 0.82], gap="small", vertical_alignment="center")
        with hero_text_col:
            st.markdown("""
            <div class="qn-eyebrow">SMART INVESTMENT SYSTEM</div>
            <h1 class="qn-headline">Make your next<br>money move <span>smarter.</span></h1>
            <p class="qn-description">A clearer view of your portfolio, market signals and long-term possibilities — all in one focused workspace.</p>
            """, unsafe_allow_html=True)
        # The bull artwork is applied as the full login-page background below.
        with hero_art_col:
            st.markdown("<div style='min-height:40px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="qn-features">
          <div class="qn-feature"><div class="qn-feature-icon">▥</div><div><b>Portfolio Intelligence</b><p>Explore allocation ideas built around your capital and risk preference.</p></div></div>
          <div class="qn-feature"><div class="qn-feature-icon">↗</div><div><b>Signals with Context</b><p>Technical indicators combined with overnight news sentiment.</p></div></div>
          <div class="qn-feature"><div class="qn-feature-icon">◇</div><div><b>Risk-Aware Investing</b><p>Scenario projections and tools to help understand potential risk.</p></div></div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="qn-workspace-card">
          <div class="qn-workspace-icon">↗</div>
          <div><div class="qn-workspace-label">YOUR INVESTMENT WORKSPACE</div>
          <div class="qn-workspace-copy">Sign in to access your personalized dashboard,<br>portfolios and market insights.</div></div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign in", "Create account"])
        with tab_login:
            st.markdown("<div class='qn-form-heading'>Welcome back</div><div class='qn-form-subtitle'>Sign in to continue to your QuantNexa dashboard.</div>", unsafe_allow_html=True)
            login_user = st.text_input("Username", key="login_usr", placeholder="Enter your username")
            login_pass = st.text_input("Password", key="login_pwd", type="password", placeholder="Enter your password")
            remember = st.checkbox("Remember me", key="login_remember")
            if st.button("→  Sign in", use_container_width=True, type="primary", key="qn_signin"):
                auth_result = database.authenticate_user(login_user, login_pass)
                if auth_result:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = auth_result
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again or create an account.")
            with st.expander("Student demo access"):
                st.caption("Use the demo account if it has been created in your database.")
                if st.button("Continue as student demo", use_container_width=True, key="qn_demo"):
                    auth_result = database.authenticate_user("student", "student123")
                    if auth_result:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = auth_result
                        st.rerun()
                    else:
                        st.error("Student demo account was not found. Please create an account or check your database.")

        with tab_signup:
            st.markdown("<div class='qn-form-heading'>Create your account</div><div class='qn-form-subtitle'>Start exploring your investment workspace.</div>", unsafe_allow_html=True)
            su_name = st.text_input("Full name", key="su_name", placeholder="Enter your full name")
            su_user = st.text_input("Username", key="su_user", placeholder="Choose a username")
            su_email = st.text_input("Email", key="su_email", placeholder="you@example.com")
            su_pass = st.text_input("Password", key="su_pass", type="password", placeholder="Create a password")
            su_type = st.selectbox("Account type", options=["Student / Secondary Income Earner", "Individual Investor", "Institutional Researcher"])
            if st.button("Create account  →", use_container_width=True, type="primary", key="qn_signup"):
                if su_user and su_email and su_pass:
                    ok, msg = database.register_user(su_user, su_name, su_email, su_pass, su_type)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter a username, email, and password.")
    st.stop()

# =========================================================================
# 5. LOGGED-IN INTERFACE
# =========================================================================
current_user = st.session_state["user"]
is_student_profile = "Student" in current_user.get("user_type", "")

nav_col1, nav_col2 = st.columns([3, 1])
with nav_col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-top: 4px;">
        <span class="neon-pill pill-purple">👤 {current_user['full_name']}</span>
        <span class="neon-pill pill-green">🏷️ {current_user['user_type']}</span>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()

# Sidebar Configuration
with st.sidebar:
    if os.path.exists("Logo.png"):
        st.image("Logo.png", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### ⚙️ **Investment Studio**")
    student_mode = st.toggle(
        "🎓 Student & Pocket-Money Mode",
        value=True if is_student_profile else False
    )

    market_choice = st.selectbox(
        "Market Domain:",
        options=["Indian Market (NSE/BSE, INR ₹)", "US / Global Market (USD $)"]
    )
    is_in_market = "Indian" in market_choice
    market_code = "IN" if is_in_market else "US"
    currency_code = "INR" if is_in_market else "USD"
    curr_sym = CURRENCY_SYMBOLS[currency_code]

    if student_mode:
        default_amt = 500.0 if is_in_market else 25.0
        min_amt = 100.0 if is_in_market else 5.0
        step_amt = 50.0 if is_in_market else 5.0
    else:
        default_amt = 25000.0 if is_in_market else 1000.0
        min_amt = 100.0 if is_in_market else 10.0
        step_amt = 500.0 if is_in_market else 50.0

    st.markdown("---")
    amount = st.number_input(
        f"Investment Capital ({curr_sym}):",
        min_value=float(min_amt),
        max_value=100000000.0,
        value=float(default_amt),
        step=float(step_amt)
    )

    risk_profile = st.select_slider(
        "Risk Appetite Strategy:",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )

    horizon_years = st.slider("Horizon (Years):", 1, 25, 5 if student_mode else 10)
    apply_sentiment = st.toggle("Live News Sentiment Tilt", value=True)

    with st.expander("🛠️ Advanced Quant Engine"):
        confidence_slider = st.select_slider("Confidence Level:", options=[0.90, 0.95, 0.98, 0.99], value=0.98)
        sim_paths = st.selectbox("Monte Carlo Paths:", options=[2000, 5000, 10000], index=1)

    st.markdown("---")
    data_source_mode = st.radio("Data Source Feed:", options=["🌐 Real-Time Live Feed", "📊 Kaggle Benchmark", "📤 Custom CSV Upload"])
    kaggle_preset = None
    uploaded_kaggle_file = None

    if "Kaggle Benchmark" in data_source_mode:
        kaggle_preset = st.selectbox("Select Dataset:", list(kaggle_data_loader.PRESET_DATASETS.keys()))
    elif "Custom CSV" in data_source_mode:
        uploaded_kaggle_file = st.file_uploader("Upload CSV:", type=["csv"])

# 6. Pipeline Execution
@st.cache_data(ttl=300, show_spinner=False)
def load_universe(m_code):
    return get_market_universe(m_code)

with st.spinner("Analyzing market prices, MPT weights, and trading signals..."):
    fallback_u = load_universe(market_code)

    if "Kaggle Benchmark" in data_source_mode and kaggle_preset:
        k_df, k_meta = kaggle_data_loader.load_preset_kaggle_dataset(kaggle_preset)
        universe = kaggle_data_loader.build_universe_from_kaggle_data(k_df, k_meta, fallback_u)
    elif "Custom CSV" in data_source_mode and uploaded_kaggle_file is not None:
        try:
            k_df, k_meta = kaggle_data_loader.parse_custom_kaggle_csv(uploaded_kaggle_file)
            universe = kaggle_data_loader.build_universe_from_kaggle_data(k_df, k_meta, fallback_u)
        except Exception:
            universe = fallback_u
    else:
        universe = fallback_u

    sample_assets = universe.get("stocks", [])[:5]
    sentiment_data = analyze_portfolio_sentiment(sample_assets)
    cagr_momentum = sentiment_data["cagr_adjustment_pct"] if apply_sentiment else 0.0

    portfolio = optimize_portfolio(
        amount=amount, currency=curr_sym, risk_profile=risk_profile,
        market=market_code, universe=universe, sentiment_adjustment=cagr_momentum
    )

    monthly_sip_sim = portfolio["sip_info"]["monthly_recommended"] if portfolio.get("sip_total", 0) > 0 else 0.0
    lump_sum_sim = portfolio["stocks_total"] + portfolio["mfs_total"]

    sim_result = run_monte_carlo_simulation(
        initial_lump_sum=lump_sum_sim, monthly_sip=monthly_sip_sim,
        expected_cagr_pct=portfolio["expected_cagr"], volatility_pct=portfolio["expected_volatility"],
        horizon_years=horizon_years, num_simulations=sim_paths, confidence_level=confidence_slider
    )

# 7. Main Header with QuantNexa Logo Banner & KPI Cards
st.markdown(f"""
<div class="qn-dashboard-hero">
  <div class="qn-dashboard-kicker">YOUR INVESTMENT COMMAND CENTER</div>
  <h1 class="qn-dashboard-title">Welcome back, <span>{current_user.get('full_name', 'Investor').split()[0]}</span></h1>
  <div class="qn-dashboard-copy">A clear view of your allocation, market signals and potential outcomes. Adjust your investment settings in the sidebar to refresh your analysis.</div>
</div>
""", unsafe_allow_html=True)

h_col1, h_col2 = st.columns([0.12, 0.88], gap="large")

with h_col1:
    if os.path.exists("Logo.png"):
        st.image("Logo.png", width=78)
    else:
        st.markdown("## ⚡")

with h_col2:
    st.markdown("""
    <div>
        <div class="hero-title" style="margin:0; line-height:1.1;">QuantNexa</div>
        <div class="hero-sub" style="margin:4px 0 0 0;">
            Smart Investment System • Markowitz MPT • Monte Carlo Scenarios • Sentiment Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='qn-section-label'>Portfolio snapshot · Based on your current settings</div>", unsafe_allow_html=True)

kpi_row1 = st.columns(3, gap="large")
kpi_row2 = st.columns(3, gap="large")
with kpi_row1[0]:
    st.markdown(f"<div class='metric-box'><div class='metric-title'>Investment capital</div><div class='metric-num'>{curr_sym} {amount:,.0f}</div><div class='metric-caption'>Capital entered in settings</div></div>", unsafe_allow_html=True)
with kpi_row1[1]:
    st.markdown(f"<div class='metric-box'><div class='metric-title'>Equities allocation</div><div class='metric-num'>{curr_sym} {portfolio['stocks_total']:,.0f}</div><div class='metric-caption'>Suggested stock allocation</div></div>", unsafe_allow_html=True)
with kpi_row1[2]:
    st.markdown(f"<div class='metric-box'><div class='metric-title'>Mutual funds</div><div class='metric-num'>{curr_sym} {portfolio['mfs_total']:,.0f}</div><div class='metric-caption'>Suggested fund allocation</div></div>", unsafe_allow_html=True)
with kpi_row2[0]:
    st.markdown(f"<div class='metric-box'><div class='metric-title'>Monthly SIP</div><div class='metric-num'>{curr_sym} {portfolio['sip_info']['monthly_recommended']:,.0f}</div><div class='metric-caption'>Suggested monthly contribution</div></div>", unsafe_allow_html=True)
with kpi_row2[1]:
    st.markdown(f"<div class='metric-box'><div class='metric-title'>Expected CAGR</div><div class='metric-num'>{portfolio['expected_cagr']}%</div><div class='metric-caption'>Modelled annualized return</div></div>", unsafe_allow_html=True)
with kpi_row2[2]:
    st.markdown(f"<div class='metric-box'><div class='metric-title'>1Y VaR (98%)</div><div class='metric-num' style='color:#FDA4AF;'>-{sim_result['var_1y_98_pct']}%</div><div class='metric-caption'>Estimated downside measure</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 8. Main Multi-Engine Tabs
st.markdown("<div class='qn-section-label'>Explore your investment tools</div>", unsafe_allow_html=True)

catalog_stocks = [s["symbol"] for s in get_catalog(market_code)]

@st.cache_data(ttl=180, show_spinner=False)
def run_coordinator_cached(sym: str, m_code: str, cap: float, strat: str):
    coord = MultiAgentCoordinator(market=m_code, capital_budget=cap)
    return coord.run_full_investigation(sym, user_capital=cap, risk_profile=strat)

tab_agent_cmd, tab_prediction, tab_strategy, tab_paper, tab_alloc, tab_mpt, tab_trading, tab_sent, tab_scenarios, tab_db, tab_guide = st.tabs([
    "🤖 Multi-Agent Command Center",
    "🔬 AI Predictions & Analogues",
    "🧪 Strategy Lab",
    "💼 Paper Trading Desk",
    "📊 Portfolio Allocation",
    "📈 Efficient Frontier",
    "⚡ Trading Signals & Risk Bot",
    "📰 24H Overnight News",
    "🔮 Wealth Scenarios",
    "📜 Saved Runs",
    "🎓 Student Guide"
])

# ----------------- TAB: MULTI-AGENT COMMAND CENTER -----------------
with tab_agent_cmd:
    st.markdown("### 🤖 **Multi-Agent Quantitative Command Center**")
    st.caption("Orchestrates 7 specialized agents (Market Data, News, Technicals, Historical Similarity, ML Forecast, Risk, and Strategy) into an explainable consolidated assessment.")

    cmd_col1, cmd_col2 = st.columns([2.5, 1])
    with cmd_col1:
        agent_target_sym = st.selectbox(
            "Select or Search Asset Symbol for Multi-Agent Investigation:",
            options=catalog_stocks,
            index=0,
            key="agent_cmd_symbol_selector"
        )
    with cmd_col2:
        custom_search_input = st.text_input("Or Enter Custom Ticker (e.g. SBIN, NVDA):", key="agent_cmd_custom_ticker")
        if custom_search_input.strip():
            agent_target_sym = normalize_symbol(custom_search_input.strip(), market_code)

    with st.spinner(f"Agents investigating {agent_target_sym}..."):
        investigation = run_coordinator_cached(agent_target_sym, market_code, amount, risk_profile)

    if investigation.get("status") == "SUCCESS":
        memo = investigation["memo"]
        agents = investigation["agents"]
        verdict = memo["final_verdict"]
        conf = memo["confidence_score"]

        pill_cls = "pill-green" if "FAVORABLE" in verdict else ("pill-rose" if "DEFENSIVE" in verdict else ("pill-amber" if "ABSTAIN" in verdict else "pill-purple"))

        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <span class="qn-eyebrow">MULTI-AGENT CONSENSUS MEMO</span>
                    <h2 style="margin:2px 0 0 0;">{memo['symbol']} &nbsp;•&nbsp; {curr_sym} {memo['last_price']:,.2f}</h2>
                    <p style="margin:4px 0 0 0; color:#94A3B8; font-size:0.88rem;">Data Freshness: {memo['data_freshness']} | Risk Regime: <b>{memo['risk_status']}</b></p>
                </div>
                <div style="text-align:right;">
                    <span class="neon-pill {pill_cls}" style="font-size:1.05rem; padding:8px 18px;">
                        🎯 {verdict}
                    </span>
                    <div style="margin-top:6px; color:#94A3B8; font-size:0.85rem;">Convergence Confidence: <b>{conf}%</b></div>
                </div>
            </div>
            <hr style="margin:14px 0; opacity:0.18;">
            <p style="margin:0; font-size:0.95rem; line-height:1.6;">
                <b>Consolidated Synthesis:</b> {memo['rationale']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        v_col1, v_col2 = st.columns([1.5, 1])
        with v_col1:
            st.plotly_chart(plot_multi_agent_consensus(memo["agent_votes"]), use_container_width=True)
        with v_col2:
            st.markdown("#### ⚖️ **Cross-Agent Consensus Matrix**")
            if memo["key_agreements"]:
                st.markdown(f"**Key Convergence:**\n- {memo['key_agreements'][0]}")
            if memo["key_conflicts"]:
                st.warning(f"⚠️ **Conflict Detected:** {memo['key_conflicts'][0]}")
            else:
                st.success("✅ **Harmony:** No major cross-agent divergence detected.")
            
            st.markdown(f"""
            <div class="metric-box" style="margin-top:12px;">
                <div class="metric-title">Actionable Protection Sizing</div>
                <div style="font-size:0.95rem; margin-top:8px;">
                    • Recommended Entry: <b>{curr_sym} {memo['actionable_levels']['recommended_entry']:,.2f}</b><br>
                    • Dynamic Stop-Loss: <b style="color:#F43F5E;">{curr_sym} {memo['actionable_levels']['dynamic_stop_loss']:,.2f}</b><br>
                    • Take-Profit Target: <b style="color:#10B981;">{curr_sym} {memo['actionable_levels']['target_1']:,.2f}</b><br>
                    • Max Planned Loss: <b style="color:#F43F5E;">{curr_sym} {memo['actionable_levels']['max_risk_amount']:,.2f}</b> ({memo['actionable_levels']['recommended_position_shares']} shares)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🧭 **Individual Agent Briefings**")
        ag_row1 = st.columns(3)
        ag_row2 = st.columns(3)

        with ag_row1[0]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Agent A: Real Market Data</div>
                <div class="metric-num">{curr_sym} {agents['market_data']['last_price']:,.2f}</div>
                <div class="metric-caption">Day: {agents['market_data']['day_change_pct']:+.2f}% • 52W: {curr_sym} {agents['market_data']['low_52w']} - {agents['market_data']['high_52w']}</div>
            </div>
            """, unsafe_allow_html=True)

        with ag_row1[1]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Agent B: News Intelligence</div>
                <div class="metric-num">{agents['news']['overall_label']}</div>
                <div class="metric-caption">Polarity: {agents['news']['composite_score']:+.2f} • {agents['news']['headlines_count']} Articles</div>
            </div>
            """, unsafe_allow_html=True)

        with ag_row1[2]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Agent C: Technical Analysis</div>
                <div class="metric-num">{agents['technical']['stance']}</div>
                <div class="metric-caption">RSI: {agents['technical']['indicators'].get('rsi', 50)} • ATR: {curr_sym} {agents['technical']['indicators'].get('atr', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with ag_row2[0]:
            sim_20 = agents['similarity'].get('horizons', {}).get('20d', {})
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Agent D: Historical Similarity</div>
                <div class="metric-num">{sim_20.get('win_rate_pct', 50)}% Win Rate</div>
                <div class="metric-caption">Median 20d: {sim_20.get('median_return_pct', 0.0):+.1f}% • {agents['similarity'].get('analogues_found', 0)} Matches</div>
            </div>
            """, unsafe_allow_html=True)

        with ag_row2[1]:
            fc_20 = agents['forecasting'].get('horizons', {}).get('20d', {})
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Agent E: Probabilistic ML</div>
                <div class="metric-num">P(+): {fc_20.get('prob_positive_pct', 50)}%</div>
                <div class="metric-caption">Brier Score: {fc_20.get('brier_score', 0.25)} • {fc_20.get('reliability', 'Normal')}</div>
            </div>
            """, unsafe_allow_html=True)

        with ag_row2[2]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Agent F: Risk Management</div>
                <div class="metric-num" style="color:#FDA4AF;">Stop: {curr_sym} {agents['risk']['stop_loss_price']}</div>
                <div class="metric-caption">Max Loss: {curr_sym} {agents['risk']['max_loss_amount']} • Vol: {agents['risk']['annualized_volatility_pct']}%</div>
            </div>
            """, unsafe_allow_html=True)

        import json as _json
        memo_json = _json.dumps(memo, indent=2)
        st.download_button(
            "📥 Export Multi-Agent Trade Memo (JSON)",
            data=memo_json,
            file_name=f"quantnexa_memo_{memo['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    else:
        st.error(investigation.get("memo", "Multi-agent analysis failed."))

# ----------------- TAB: AI PREDICTIONS & HISTORICAL ANALOGUES -----------------
with tab_prediction:
    st.markdown("### 🔬 **AI Probabilistic Outcome Engine & Historical Pattern Analogues**")
    st.caption("Matches multi-factor market situations with verified historical episodes. Predicts calibrated out-of-sample forward distributions without lookahead bias.")

    pred_col1, pred_col2 = st.columns([2, 1])
    with pred_col1:
        pred_sym = st.selectbox("Select Asset for Prediction Analysis:", options=catalog_stocks, index=0, key="pred_tab_sym_select")
    with pred_col2:
        pred_horizon = st.radio("Forecast Horizon:", options=["1d", "5d", "20d"], format_func=lambda x: "Next 1 Day (1d)" if x == "1d" else ("Next 5 Days (5d)" if x == "5d" else "Next 20 Days (20d)"), horizontal=True)

    with st.spinner("Extracting historical patterns and validating models..."):
        p_coord = run_coordinator_cached(pred_sym, market_code, amount, risk_profile)

    if p_coord.get("status") == "SUCCESS":
        sim_data = p_coord["agents"]["similarity"]
        fc_data = p_coord["agents"]["forecasting"]
        h_data = sim_data.get("horizons", {}).get(pred_horizon, {})
        f_data = fc_data.get("horizons", {}).get(pred_horizon, {})

        p_row1, p_row2 = st.columns(2)
        with p_row1:
            st.plotly_chart(plot_forecast_probabilities(fc_data.get("horizons", {})), use_container_width=True)
        with p_row2:
            st.plotly_chart(plot_similarity_distribution(sim_data.get("horizons", {}), pred_horizon), use_container_width=True)

        st.markdown(f"#### 📊 **Empirical Outcome Scorecard ({pred_horizon} Horizon)**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Win Rate (Historical)</div><div class='metric-num' style='color:#10B981;'>{h_data.get('win_rate_pct', 50)}%</div><div class='metric-caption'>{h_data.get('sample_count', 0)} Analogue Samples</div></div>", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Median Return</div><div class='metric-num'>{h_data.get('median_return_pct', 0.0):+.2f}%</div><div class='metric-caption'>IQR: [{h_data.get('p25_pct', 0):+.1f}%, {h_data.get('p75_pct', 0):+.1f}%]</div></div>", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>ML P(Gain) Probability</div><div class='metric-num' style='color:#38BDF8;'>{f_data.get('prob_positive_pct', 50)}%</div><div class='metric-caption'>Brier Score: {f_data.get('brier_score', 0.25)}</div></div>", unsafe_allow_html=True)
        with sc4:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Downside Tail Risk</div><div class='metric-num' style='color:#FDA4AF;'>{h_data.get('worst_case_pct', 0.0):+.2f}%</div><div class='metric-caption'>Best Historical: {h_data.get('best_case_pct', 0.0):+.2f}%</div></div>", unsafe_allow_html=True)

        st.markdown("#### 📜 **Matched Historical Analogues & Forward Performance**")
        if sim_data.get("analogues"):
            a_df = pd.DataFrame(sim_data["analogues"])
            display_cols = ["date", "similarity", "rsi", "ret_20d", f"fwd_{pred_horizon}_ret"]
            avail_cols = [c for c in display_cols if c in a_df.columns]
            a_df_renamed = a_df[avail_cols].rename(columns={
                "date": "Historical Date",
                "similarity": "Similarity Score (%)",
                "rsi": "RSI at Time",
                "ret_20d": "Prior 20D Return (%)",
                f"fwd_{pred_horizon}_ret": f"Subsequent {pred_horizon} Return (%)"
            })
            st.dataframe(a_df_renamed, use_container_width=True, hide_index=True)
        else:
            st.info("No matching analogues found.")

        st.info(f"💡 {sim_data.get('disclaimer', '')}")

# ----------------- TAB: STRATEGY LAB -----------------
with tab_strategy:
    st.markdown("### 🧪 **Quantitative Strategy Lab & Backtesting Engine**")
    st.caption("Backtest algorithmic trading rules across historical data with realistic transaction costs, slippage, and equity curves.")

    s_c1, s_c2, s_c3 = st.columns([1.5, 1.5, 1])
    with s_c1:
        strat_sym = st.selectbox("Select Asset for Strategy Testing:", options=catalog_stocks, index=0, key="strat_tab_sym_select")
    with s_c2:
        selected_strategy = st.selectbox(
            "Select Quantitative Strategy:",
            options=[
                "EMA Trend Following (EMA 20/50 Cross)",
                "RSI Mean Reversion (RSI < 32 Entry, > 62 Exit)",
                "Momentum Breakout (20-Day Donchian)",
                "Bollinger Band Mean Reversion"
            ]
        )
    with s_c3:
        strat_capital = st.number_input(
    "Backtest Capital:",
    min_value=1000.0,
    max_value=10000000.0,
    value=max(1000.0, float(amount)),
    step=5000.0
)
    strat_agent = QuantitativeStrategyAgent(risk_free_rate=0.05, transaction_cost_pct=0.10)
    data_agent = MarketDataAgent(market=market_code)
    hist_ohlcv = data_agent.fetch_ohlcv(strat_sym, period="3y").get("df", pd.DataFrame())

    if not hist_ohlcv.empty:
        strat_result = strat_agent.backtest(hist_ohlcv, strategy_name=selected_strategy, initial_capital=strat_capital)

        sr1, sr2, sr3, sr4, sr5, sr6 = st.columns(6)
        with sr1:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Total Return</div><div class='metric-num' style='color:#10B981;'>{strat_result['total_return_pct']:+.1f}%</div><div class='metric-caption'>Buy&Hold: {strat_result['benchmark_return_pct']:+.1f}%</div></div>", unsafe_allow_html=True)
        with sr2:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Strategy CAGR</div><div class='metric-num'>{strat_result['cagr_pct']:.1f}%</div><div class='metric-caption'>Bench: {strat_result['benchmark_cagr_pct']:.1f}%</div></div>", unsafe_allow_html=True)
        with sr3:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Max Drawdown</div><div class='metric-num' style='color:#FDA4AF;'>{strat_result['max_drawdown_pct']:.1f}%</div><div class='metric-caption'>Bench: {strat_result['benchmark_max_drawdown_pct']:.1f}%</div></div>", unsafe_allow_html=True)
        with sr4:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Sharpe Ratio</div><div class='metric-num'>{strat_result['sharpe_ratio']:.2f}</div><div class='metric-caption'>Risk-Free: 5.0%</div></div>", unsafe_allow_html=True)
        with sr5:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Win Rate</div><div class='metric-num'>{strat_result['win_rate_pct']}%</div><div class='metric-caption'>{strat_result['total_trades']} Trades Executed</div></div>", unsafe_allow_html=True)
        with sr6:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Profit Factor</div><div class='metric-num'>{strat_result['profit_factor']:.2f}</div><div class='metric-caption'>Cost: 0.10%/trade</div></div>", unsafe_allow_html=True)

        st.plotly_chart(plot_strategy_equity_curve(strat_result["equity_df"], selected_strategy, curr_sym), use_container_width=True)
        st.info(f"📋 **Strategy Audit Memo:** {strat_result['summary']}")
    else:
        st.error(f"Unable to load historical data for {strat_sym}.")

# ----------------- TAB: PAPER TRADING DESK -----------------
with tab_paper:
    st.markdown("### 💼 **QuantNexa Paper Trading Desk & Simulated Execution**")
    st.caption("Practice executing multi-agent signals with virtual capital. Tracks real-time positions, cash, and P&L with strict safety controls.")

    st.warning("⚠️ **Safety Notice:** Paper Trading Desk is strictly simulated. Live broker execution is disabled by default. No real funds are at risk.")

    paper_acc = database.get_or_create_paper_account(current_user["username"], currency_code, 100000.0)

    open_pos_df = database.get_paper_open_positions(current_user["username"])
    total_invested_paper = open_pos_df["Invested"].sum() if not open_pos_df.empty else 0.0
    total_paper_equity = paper_acc["cash_balance"] + total_invested_paper

    pk1, pk2, pk3, pk4 = st.columns(4)
    with pk1:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Virtual Cash Balance</div><div class='metric-num'>{curr_sym} {paper_acc['cash_balance']:,.2f}</div><div class='metric-caption'>Available Buying Power</div></div>", unsafe_allow_html=True)
    with pk2:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Active Deployed Capital</div><div class='metric-num'>{curr_sym} {total_invested_paper:,.2f}</div><div class='metric-caption'>{len(open_pos_df)} Open Positions</div></div>", unsafe_allow_html=True)
    with pk3:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Total Virtual Portfolio</div><div class='metric-num' style='color:#38BDF8;'>{curr_sym} {total_paper_equity:,.2f}</div><div class='metric-caption'>Cash + Deployed Equities</div></div>", unsafe_allow_html=True)
    with pk4:
        paper_return_pct = ((total_paper_equity - 100000.0) / 100000.0) * 100.0
        ret_color = "#10B981" if paper_return_pct >= 0 else "#FDA4AF"
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Cumulative Paper P&L</div><div class='metric-num' style='color:{ret_color};'>{paper_return_pct:+.2f}%</div><div class='metric-caption'>Base: {curr_sym} 100,000.00</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚡ **Simulated Order Placement Ticket**")

    ord_c1, ord_c2, ord_c3, ord_c4 = st.columns(4)
    with ord_c1:
        order_sym = st.selectbox("Order Symbol:", options=catalog_stocks, index=0, key="paper_order_sym_select")
    with ord_c2:
        order_side = st.selectbox("Order Side:", options=["BUY", "SELL"], key="paper_order_side_select")
    with ord_c3:
        live_q = MarketDataAgent(market=market_code).get_quote(order_sym)
        raw_price = float(live_q.get("last_price", 100.0) or 100.0)
        safe_price = max(raw_price, 1.0)
        order_px = st.number_input("Limit / Market Price:", min_value=1.0, value=safe_price, step=1.0)
    with ord_c4:
        order_qty = st.number_input("Quantity (Shares):", min_value=1, max_value=10000, value=10, step=1)

    ord_s1, ord_s2, ord_s3 = st.columns([1, 1, 2])
    with ord_s1:
        order_sl = st.number_input("Stop-Loss Price:", min_value=0.0, value=round(order_px * 0.95, 2), step=1.0)
    with ord_s2:
        order_tp = st.number_input("Target Price:", min_value=0.0, value=round(order_px * 1.10, 2), step=1.0)
    with ord_s3:
        order_reason = st.text_input("Trade Evidence / Model Rationale:", value="QuantNexa Multi-Agent Signal Consensus")

    total_order_cost = order_qty * order_px
    st.caption(f"Estimated Order Value: **{curr_sym} {total_order_cost:,.2f}**")

    confirm_trade = st.checkbox("☑️ Explicit User Confirmation: I confirm executing this simulated paper order.", key="paper_confirm_checkbox")
    if st.button("🚀 Transmit Simulated Order", type="primary", use_container_width=True):
        if not confirm_trade:
            st.error("Please tick the explicit confirmation box before submitting simulated orders.")
        else:
            success, msg = database.execute_paper_order(
                username=current_user["username"],
                symbol=order_sym,
                side=order_side,
                quantity=order_qty,
                price=order_px,
                order_type="MARKET",
                stop_loss=order_sl,
                target_price=order_tp,
                evidence_notes=order_reason
            )
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("#### 📂 **Active Open Paper Positions**")
    if not open_pos_df.empty:
        st.dataframe(open_pos_df, use_container_width=True, hide_index=True)
        close_c1, close_c2 = st.columns([2, 1])
        with close_c1:
            close_trade_id = st.selectbox("Select Trade ID to Close:", options=open_pos_df["ID"].tolist())
        with close_c2:
            if st.button("Close Position (Market Price)", use_container_width=True):
                row_selected = open_pos_df[open_pos_df["ID"] == close_trade_id].iloc[0]
                latest_mkt_q = MarketDataAgent(market=market_code).get_quote(row_selected["Symbol"])
                c_px = latest_mkt_q.get("last_price", row_selected["Entry Price"])
                ok, c_msg = database.close_paper_position(close_trade_id, c_px, current_user["username"])
                if ok:
                    st.success(c_msg)
                    st.rerun()
                else:
                    st.error(c_msg)
    else:
        st.info("No active open paper trading positions.")

    st.markdown("#### 📜 **Paper Execution Audit Log**")
    p_hist = database.get_paper_history(current_user["username"])
    if not p_hist.empty:
        st.dataframe(p_hist, use_container_width=True, hide_index=True)

# ----------------- TAB 5: ALLOCATION -----------------
with tab_alloc:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_asset_allocation(portfolio, curr_sym), use_container_width=True)
    with c2:
        st.plotly_chart(plot_holdings_breakup(portfolio, curr_sym), use_container_width=True)

# ----------------- TAB 2: MPT FRONTIER -----------------
with tab_mpt:
    ef_data = portfolio.get("efficient_frontier")
    if ef_data:
        st.plotly_chart(plot_efficient_frontier(ef_data, portfolio["optimal_point"], portfolio["gmv_point"]), use_container_width=True)

# ----------------- TAB 3: LIVE TRADING SIGNALS & RISK ANALYSIS -----------------
with tab_trading:
    st.markdown("### ⚡ **Real-Time Trading Signals, Stop-Loss & Risk Engine**")
    
    trader_mode = st.radio(
        "Analysis View:",
        options=["🛡️ Beginner Saver Protection Mode", "📊 Pro Trader Technical & Options Mode"],
        horizontal=True
    )
    is_pro_mode = "Pro Trader" in trader_mode

    selected_ticker = st.selectbox(
        "Select Asset Ticker for Analysis:",
        options=[s["symbol"] for s in universe.get("stocks", [])],
        index=0
    )

    tech_data = analyze_technical_indicators(selected_ticker)
    news_data = harvest_overnight_news(selected_ticker)
    final_signal = reconcile_signal_with_news(tech_data["signal"], news_data["composite_score"])
    
    risk_data = calculate_trade_risk(
        entry_price=tech_data["last_price"],
        atr_value=tech_data["atr"],
        total_savings=amount,
        currency_symbol=curr_sym
    )

    pill_style = "pill-green" if "BUY" in final_signal else ("pill-rose" if "SELL" in final_signal else "pill-amber")
    st.markdown(f"""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h3 style="margin:0;">{selected_ticker} Engine Reading</h3>
                <p style="margin:4px 0 0 0; color:#94A3B8;">Price: <b>{curr_sym} {tech_data['last_price']:,.2f}</b></p>
            </div>
            <div>
                <span class="neon-pill {pill_style}" style="font-size:1.1rem; padding:8px 18px;">
                    🎯 Consensus: {final_signal}
                </span>
            </div>
        </div>
        <hr style="margin:14px 0; opacity:0.2;">
        <p style="margin:0; font-size:0.92rem;">
            <b>Technical Summary:</b> {tech_data['summary']}<br>
            <b>Overnight News Pulse:</b> {news_data['summary']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>🛑 Dynamic Stop-Loss</div><div class='metric-num' style='color:#F43F5E;'>{curr_sym} {risk_data['stop_loss_price']:,.2f}</div><div class='metric-caption'>1.5x ATR Protection</div></div>", unsafe_allow_html=True)
    with r2:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>🎯 Target Price</div><div class='metric-num' style='color:#10B981;'>{curr_sym} {risk_data['target_price']:,.2f}</div><div class='metric-caption'>1:2 Risk-Reward Ratio</div></div>", unsafe_allow_html=True)
    with r3:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>💼 Safe Position</div><div class='metric-num'>{risk_data['recommended_units']} Units</div><div class='metric-caption'>Cost: {curr_sym} {risk_data['position_value']:,.2f}</div></div>", unsafe_allow_html=True)
    with r4:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>⚠️ Max Capital Risk</div><div class='metric-num' style='color:#F43F5E;'>{curr_sym} {risk_data['max_loss_amount']:,.2f}</div><div class='metric-caption'>Capped at {risk_data['max_risk_pct']}%</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if is_pro_mode:
        if not tech_data["df_ohlc"].empty:
            st.plotly_chart(plot_candlestick_trading(tech_data["df_ohlc"], selected_ticker, curr_sym), use_container_width=True)

        options_res = simulate_options_payoff(
            stock_price=tech_data["last_price"], strike_price=tech_data["last_price"]
        )
        o1, o2 = st.columns(2)
        with o1:
            st.plotly_chart(plot_options_payoff(options_res["payoff_df"], curr_sym), use_container_width=True)
        with o2:
            st.markdown("##### 📐 Black-Scholes Call/Put Options Greeks")
            st.json({
                "Call Premium": f"{curr_sym} {options_res['call_premium']}",
                "Put Premium": f"{curr_sym} {options_res['put_premium']}",
                "Delta (Δ)": options_res["call_greeks"]["delta"],
                "Gamma (Γ)": options_res["call_greeks"]["gamma"],
                "Theta (Θ)": options_res["call_greeks"]["theta"],
                "Vega (V)": options_res["call_greeks"]["vega"]
            })
    else:
        st.info(f"🛡️ **Capital Preservation Advice**: {risk_data['risk_summary']}")

# ----------------- TAB 4: OVERNIGHT NEWS -----------------
with tab_sent:
    st.markdown("### 📰 **24-Hour Financial News Pulse & Sentiment Harvester**")

    news_ticker = st.selectbox(
        "Select Asset Ticker for Overnight News Analysis:",
        options=[s["symbol"] for s in universe.get("stocks", [])],
        index=0,
        key="news_tab_ticker_select"
    )

    live_news_data = harvest_overnight_news(news_ticker)

    n_col1, n_col2 = st.columns([1.2, 1.8])

    with n_col1:
        st.plotly_chart(
            plot_sentiment_gauge(live_news_data["composite_score"], live_news_data["overall_label"]),
            use_container_width=True
        )

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Overnight Polarity Score</div>
            <div class="metric-num">{live_news_data['composite_score']:+.2f}</div>
            <div class="metric-caption">Scraped Headlines Count: {live_news_data['headlines_count']}</div>
        </div>
        """, unsafe_allow_html=True)

    with n_col2:
        st.markdown(f"#### 📄 Scraped Headlines for **{news_ticker}**")

        if live_news_data["headlines"]:
            news_df = pd.DataFrame(live_news_data["headlines"])
            formatted_df = news_df[["title", "source", "sentiment", "compound"]].rename(columns={
                "title": "Headline",
                "source": "News Source",
                "sentiment": "Sentiment Tag",
                "compound": "Polarity Score"
            })

            st.dataframe(
                formatted_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Headline": st.column_config.TextColumn("Headline", width="medium"),
                    "News Source": st.column_config.TextColumn("News Source", width="small"),
                    "Sentiment Tag": st.column_config.TextColumn("Sentiment Tag", width="small"),
                    "Polarity Score": st.column_config.NumberColumn("Polarity Score", format="%.2f")
                }
            )
        else:
            st.info(f"No recent news articles found for {news_ticker} in the last 24 hours. Sentiment defaults to Neutral.")

# ----------------- TAB 5: SCENARIOS -----------------
with tab_scenarios:
    st.plotly_chart(plot_scenario_projections(sim_result, curr_sym), use_container_width=True)
    st.plotly_chart(plot_wealth_comparison_bars(sim_result, curr_sym), use_container_width=True)

# ----------------- TAB 6: SAVED PORTFOLIOS -----------------
with tab_db:
    user_db_df = database.get_all_portfolios(username=current_user["username"])
    if not user_db_df.empty:
        st.dataframe(user_db_df, use_container_width=True, hide_index=True)
    else:
        st.info("No saved portfolio runs found in the database yet.")

# ----------------- TAB 7: STUDENT GUIDE -----------------
with tab_guide:
    st.markdown("### 🎓 **QuantNexa Micro-Investing Blueprint**")
    
    st.info("""
    💡 **Micro-Investing & Capital Growth Plan**
    
    By automating allocations as low as **₹100 to ₹500 ($5 to $10)** per month, students and early earners harness the power of **Compounding** while mitigating drawdown risks through index diversification.
    """)

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("#### 🚀 Starter Roadmap")
        st.markdown("""
        * **Open Zero-AMC Demat Account:** Select discount brokers offering zero annual fee accounts for students.
        * **Automate SIP Reserve:** Set auto-debit on the 1st of every month right after receiving your stipend or allowance.
        * **Index First Strategy:** Allocate 80% of capital to low-cost Nifty 50 or S&P 500 Index funds (Expense Ratio < 0.10%).
        * **Avoid Derivative Traps:** Steer clear of short-term options trading calls/puts to avoid Theta time decay.
        """)

    with g_col2:
        st.markdown("#### 🛡️ Risk & Allocation Rules")
        st.markdown("""
        * **1.5% Single-Stock Cap:** Never allocate more than 1.5% of total liquid savings to a single stock position.
        * **Emergency Reserve First:** Keep 3 months of essential living expenses in a liquid high-yield savings account.
        * **Dollar-Cost Averaging (DCA):** Treat market drawdowns as long-term buying opportunities over 5+ year horizons.
        """)

    st.markdown("---")
    st.markdown("#### 🧮 Interactive Compound SIP Growth Estimator")
    
    sip_input = st.slider("Select Monthly Micro-SIP Amount:", min_value=100, max_value=2000, value=500, step=100)
    years_input = st.slider("Investment Horizon (Years):", min_value=1, max_value=15, value=5)
    cagr_est = 13.5

    monthly_rate = (cagr_est / 100) / 12
    total_months = years_input * 12
    future_val = sip_input * (((1 + monthly_rate)**total_months - 1) / monthly_rate) * (1 + monthly_rate)
    total_invested_sip = sip_input * total_months
    wealth_gained = future_val - total_invested_sip

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Invested Capital</div><div class='metric-num'>{curr_sym} {total_invested_sip:,.0f}</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Projected Value</div><div class='metric-num' style='color:#10B981;'>{curr_sym} {future_val:,.0f}</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Est. Net Profit</div><div class='metric-num' style='color:#38BDF8;'>+{curr_sym} {wealth_gained:,.0f}</div></div>", unsafe_allow_html=True)