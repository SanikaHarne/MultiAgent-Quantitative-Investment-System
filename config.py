"""
Configuration and Asset Catalog for Smart Analysis & Investment System.
Supports Indian Market (NSE/BSE, INR ₹) and US/Global Market (USD $).
"""

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$"
}

# Risk profile asset allocation presets (Stocks %, Mutual Funds %, SIP %)
ALLOCATION_PROFILES = {
    "Conservative": {
        "description": "Prioritizes capital preservation with steady, lower-volatility compounding.",
        "stocks_pct": 20.0,
        "mutual_funds_pct": 50.0,
        "sip_pct": 30.0,
        "expected_cagr_inr": 11.5,
        "volatility_inr": 10.0,
        "expected_cagr_usd": 8.5,
        "volatility_usd": 9.0
    },
    "Moderate": {
        "description": "Balanced approach between capital appreciation and market resilience.",
        "stocks_pct": 40.0,
        "mutual_funds_pct": 35.0,
        "sip_pct": 25.0,
        "expected_cagr_inr": 14.5,
        "volatility_inr": 15.0,
        "expected_cagr_usd": 11.0,
        "volatility_usd": 14.0
    },
    "Aggressive": {
        "description": "Maximizes long-term wealth creation by overweighting high-growth equities.",
        "stocks_pct": 60.0,
        "mutual_funds_pct": 25.0,
        "sip_pct": 15.0,
        "expected_cagr_inr": 17.5,
        "volatility_inr": 21.0,
        "expected_cagr_usd": 14.5,
        "volatility_usd": 19.5
    }
}

# Catalog of vetted quality assets for Indian Market (INR)
ASSETS_IN = {
    "stocks": [
        {
            "symbol": "RELIANCE.NS",
            "name": "Reliance Industries Ltd",
            "sector": "Energy & Conglomerate",
            "approx_price": 2980.0,
            "cagr_5y": 17.2,
            "volatility": 18.5,
            "pe_ratio": 26.4,
            "category": "Large Cap / Bluechip"
        },
        {
            "symbol": "TCS.NS",
            "name": "Tata Consultancy Services",
            "sector": "Information Technology",
            "approx_price": 4250.0,
            "cagr_5y": 15.8,
            "volatility": 16.2,
            "pe_ratio": 30.1,
            "category": "Large Cap / Bluechip"
        },
        {
            "symbol": "HDFCBANK.NS",
            "name": "HDFC Bank Ltd",
            "sector": "Financial Services",
            "approx_price": 1650.0,
            "cagr_5y": 14.0,
            "volatility": 17.0,
            "pe_ratio": 19.2,
            "category": "Large Cap / Bluechip"
        },
        {
            "symbol": "INFY.NS",
            "name": "Infosys Ltd",
            "sector": "Information Technology",
            "approx_price": 1890.0,
            "cagr_5y": 16.5,
            "volatility": 19.4,
            "pe_ratio": 27.5,
            "category": "Large Cap / Bluechip"
        },
        {
            "symbol": "ICICIBANK.NS",
            "name": "ICICI Bank Ltd",
            "sector": "Financial Services",
            "approx_price": 1280.0,
            "cagr_5y": 21.4,
            "volatility": 18.2,
            "pe_ratio": 18.5,
            "category": "Large Cap / Growth"
        },
        {
            "symbol": "BAJFINANCE.NS",
            "name": "Bajaj Finance Ltd",
            "sector": "Financial Services & NBFC",
            "approx_price": 7150.0,
            "cagr_5y": 20.5,
            "volatility": 22.0,
            "pe_ratio": 28.5,
            "category": "Large Cap / High Growth"
        },
        {
            "symbol": "ITC.NS",
            "name": "ITC Ltd",
            "sector": "FMCG & Conglomerate",
            "approx_price": 495.0,
            "cagr_5y": 18.2,
            "volatility": 14.8,
            "pe_ratio": 27.8,
            "category": "Dividend & Value"
        },
        {
            "symbol": "BHARTIARTL.NS",
            "name": "Bharti Airtel Ltd",
            "sector": "Telecom & Digital",
            "approx_price": 1680.0,
            "cagr_5y": 24.1,
            "volatility": 19.0,
            "pe_ratio": 45.0,
            "category": "Large Cap / Telecom"
        },
        {
            "symbol": "LT.NS",
            "name": "Larsen & Toubro Ltd",
            "sector": "Infrastructure & Capital Goods",
            "approx_price": 3620.0,
            "cagr_5y": 22.0,
            "volatility": 20.1,
            "pe_ratio": 34.0,
            "category": "Large Cap / Infra"
        },
        {
            "symbol": "HINDUNILVR.NS",
            "name": "Hindustan Unilever Ltd",
            "sector": "FMCG Defensives",
            "approx_price": 2800.0,
            "cagr_5y": 11.2,
            "volatility": 13.5,
            "pe_ratio": 54.0,
            "category": "Defensive / Dividend"
        },
        {
            "symbol": "NIFTYBEES.NS",
            "name": "Nippon India ETF Nifty 50 BeES",
            "sector": "Index ETF (India Top 50)",
            "approx_price": 285.0,
            "cagr_5y": 15.6,
            "volatility": 14.2,
            "pe_ratio": 22.0,
            "category": "Student Micro-Asset (₹285)"
        },
        {
            "symbol": "GOLDBEES.NS",
            "name": "Nippon India ETF Gold BeES",
            "sector": "Commodity / Sovereign Gold Hedge",
            "approx_price": 68.0,
            "cagr_5y": 13.8,
            "volatility": 11.0,
            "pe_ratio": 1.0,
            "category": "Pocket Money Micro-Asset (₹68)"
        },
        {
            "symbol": "TATASTEEL.NS",
            "name": "Tata Steel Ltd",
            "sector": "Metals & Mining",
            "approx_price": 160.0,
            "cagr_5y": 18.2,
            "volatility": 24.5,
            "pe_ratio": 14.0,
            "category": "Affordable Stock (₹160)"
        },
        {
            "symbol": "IOC.NS",
            "name": "Indian Oil Corporation Ltd",
            "sector": "Energy & High Dividend PSU",
            "approx_price": 140.0,
            "cagr_5y": 16.5,
            "volatility": 19.0,
            "pe_ratio": 9.5,
            "category": "High Dividend Stock (₹140)"
        }
    ],
    "mutual_funds": [
        {
            "scheme_code": "148726",
            "name": "Navi Nifty 50 Index Fund - Direct Plan - Growth",
            "category": "Large Cap / ₹100 Student SIP",
            "cagr_5y": 15.8,
            "volatility": 14.1,
            "expense_ratio": 0.06,
            "min_lumpsum": 100,
            "min_sip": 100
        },
        {
            "scheme_code": "120503",
            "name": "ICICI Prudential Nifty 50 Index Fund - Direct Plan - Growth",
            "category": "Large Cap / ₹100 Student SIP",
            "cagr_5y": 15.7,
            "volatility": 14.2,
            "expense_ratio": 0.17,
            "min_lumpsum": 100,
            "min_sip": 100
        },
        {
            "scheme_code": "120716",
            "name": "UTI Nifty 50 Index Fund - Direct Plan - Growth",
            "category": "Large Cap / Index Fund",
            "cagr_5y": 15.6,
            "volatility": 14.2,
            "expense_ratio": 0.18,
            "min_lumpsum": 250,
            "min_sip": 250
        },
        {
            "scheme_code": "122639",
            "name": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
            "category": "Flexi Cap / Global Equity",
            "cagr_5y": 20.8,
            "volatility": 13.8,
            "expense_ratio": 0.63,
            "min_lumpsum": 1000,
            "min_sip": 1000
        },
        {
            "scheme_code": "118834",
            "name": "Mirae Asset Large & Midcap Fund - Direct Plan - Growth",
            "category": "Large & Midcap",
            "cagr_5y": 18.9,
            "volatility": 16.5,
            "expense_ratio": 0.58,
            "min_lumpsum": 1000,
            "min_sip": 500
        },
        {
            "scheme_code": "118989",
            "name": "HDFC Balanced Advantage Fund - Direct Plan - Growth",
            "category": "Dynamic Asset Allocation / Hybrid",
            "cagr_5y": 16.4,
            "volatility": 11.5,
            "expense_ratio": 0.72,
            "min_lumpsum": 500,
            "min_sip": 500
        },
        {
            "scheme_code": "125497",
            "name": "Nippon India Small Cap Fund - Direct Plan - Growth",
            "category": "Small Cap High Growth",
            "cagr_5y": 27.5,
            "volatility": 21.0,
            "expense_ratio": 0.70,
            "min_lumpsum": 5000,
            "min_sip": 1000
        }
    ]
}

# Catalog of vetted quality assets for US/Global Market (USD)
ASSETS_US = {
    "stocks": [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "sector": "Technology & Consumer Devices",
            "approx_price": 225.0,
            "cagr_5y": 18.5,
            "volatility": 20.1,
            "pe_ratio": 33.5,
            "category": "Mega Cap / Tech"
        },
        {
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "sector": "Enterprise Software & Cloud AI",
            "approx_price": 430.0,
            "cagr_5y": 20.2,
            "volatility": 21.5,
            "pe_ratio": 35.2,
            "category": "Mega Cap / Cloud"
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc. (Google)",
            "sector": "Internet & Search Services",
            "approx_price": 178.0,
            "cagr_5y": 19.0,
            "volatility": 23.0,
            "pe_ratio": 24.1,
            "category": "Mega Cap / AI & Ads"
        },
        {
            "symbol": "AMZN",
            "name": "Amazon.com Inc.",
            "sector": "E-Commerce & AWS Cloud",
            "approx_price": 185.0,
            "cagr_5y": 16.8,
            "volatility": 26.0,
            "pe_ratio": 41.0,
            "category": "Consumer Cyclical & Cloud"
        },
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation",
            "sector": "Semiconductors & AI Compute",
            "approx_price": 125.0,
            "cagr_5y": 55.0,
            "volatility": 42.0,
            "pe_ratio": 46.0,
            "category": "High Growth / Semiconductor"
        },
        {
            "symbol": "BRK-B",
            "name": "Berkshire Hathaway Inc.",
            "sector": "Conglomerate & Value",
            "approx_price": 450.0,
            "cagr_5y": 14.2,
            "volatility": 15.0,
            "pe_ratio": 20.5,
            "category": "Value / Defensive"
        },
        {
            "symbol": "JPM",
            "name": "JPMorgan Chase & Co.",
            "sector": "Financial & Banking",
            "approx_price": 210.0,
            "cagr_5y": 15.5,
            "volatility": 18.0,
            "pe_ratio": 12.0,
            "category": "Financial Services"
        },
        {
            "symbol": "V",
            "name": "Visa Inc.",
            "sector": "Payments & Fintech",
            "approx_price": 275.0,
            "cagr_5y": 13.8,
            "volatility": 17.5,
            "pe_ratio": 29.0,
            "category": "Global Payments"
        }
    ],
    "mutual_funds": [
        {
            "scheme_code": "VOO",
            "name": "Vanguard S&P 500 ETF (Index Fund)",
            "category": "US Large Cap Core",
            "cagr_5y": 14.8,
            "volatility": 15.2,
            "expense_ratio": 0.03,
            "min_lumpsum": 50,
            "min_sip": 25
        },
        {
            "scheme_code": "QQQ",
            "name": "Invesco QQQ Trust (Nasdaq 100 ETF)",
            "category": "Tech & Growth Index",
            "cagr_5y": 18.9,
            "volatility": 20.5,
            "expense_ratio": 0.20,
            "min_lumpsum": 50,
            "min_sip": 25
        },
        {
            "scheme_code": "VT",
            "name": "Vanguard Total World Stock ETF",
            "category": "Global All-Cap Diversified",
            "cagr_5y": 11.5,
            "volatility": 14.5,
            "expense_ratio": 0.07,
            "min_lumpsum": 50,
            "min_sip": 25
        },
        {
            "scheme_code": "SCHD",
            "name": "Schwab U.S. Dividend Equity ETF",
            "category": "High Dividend / Quality Value",
            "cagr_5y": 12.8,
            "volatility": 13.8,
            "expense_ratio": 0.06,
            "min_lumpsum": 50,
            "min_sip": 25
        },
        {
            "scheme_code": "BND",
            "name": "Vanguard Total Bond Market ETF",
            "category": "Fixed Income / Low Risk Bond",
            "cagr_5y": 3.5,
            "volatility": 6.5,
            "expense_ratio": 0.03,
            "min_lumpsum": 50,
            "min_sip": 25
        }
    ]
}
