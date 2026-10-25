"""
Database Persistence & User Authentication Module:
Manages SQLite via SQLAlchemy for User Accounts (Signup/Login)
and persists customized portfolios linked to specific users.
"""

import os
import json
import hashlib
import binascii
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    desc
)
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "investments.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def hash_password(password: str) -> str:
    """Creates a secure salted SHA-256 hash."""
    salt = hashlib.sha256(b"smart_invest_salt_2026").digest()
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return binascii.hexlify(key).decode("utf-8")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    user_type = Column(String(50), default="Student / Micro-Investor")
    created_at = Column(DateTime, default=datetime.utcnow)


class PortfolioHistory(Base):
    __tablename__ = "portfolio_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    username = Column(String(50), index=True, default="guest")
    client_name = Column(String(100), default="Individual Investor")
    currency = Column(String(10), default="INR")
    total_capital = Column(Float, nullable=False)
    risk_profile = Column(String(50), nullable=False)
    horizon_years = Column(Integer, nullable=False)
    
    # Financial metrics
    expected_cagr = Column(Float, nullable=False)
    expected_volatility = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    
    # Sentiment metrics
    sentiment_score = Column(Float, default=0.0)
    sentiment_label = Column(String(50), default="Neutral")
    
    # Scenario metrics
    worst_case_wealth = Column(Float, nullable=False)
    average_case_wealth = Column(Float, nullable=False)
    best_case_wealth = Column(Float, nullable=False)
    var_98_pct = Column(Float, nullable=False)
    
    # Full JSON serializations
    holdings_json = Column(Text, default="{}")


class PaperAccount(Base):
    __tablename__ = "paper_accounts"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    currency = Column(String(10), default="INR")
    cash_balance = Column(Float, default=100000.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    username = Column(String(50), index=True, nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # BUY or SELL
    order_type = Column(String(20), default="MARKET")
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, default=0.0)
    target_price = Column(Float, default=0.0)
    status = Column(String(20), default="OPEN")  # OPEN, CLOSED, STOPPED_OUT, TARGET_REACHED
    exit_price = Column(Float, default=0.0)
    exit_timestamp = Column(DateTime, nullable=True)
    realized_pnl = Column(Float, default=0.0)
    evidence_notes = Column(Text, default="")
    model_version = Column(String(50), default="Multi-Agent v2.0")


def _migrate_tables():
    """
    Auto-migration: detects columns present in the ORM models but missing
    from the existing SQLite table and ALTERs the table to add them.
    This prevents OperationalError when the schema evolves without deleting the DB.
    """
    import sqlalchemy as _sa
    with engine.connect() as conn:
        for table in Base.metadata.sorted_tables:
            try:
                result = conn.execute(_sa.text(f"PRAGMA table_info({table.name})"))
                existing_cols = {row[1] for row in result.fetchall()}
                for col in table.columns:
                    if col.name not in existing_cols:
                        col_type = col.type.compile(dialect=engine.dialect)
                        default_clause = ""
                        if col.default is not None and col.default.is_scalar:
                            val = col.default.arg
                            default_clause = (
                                f" DEFAULT '{val}'" if isinstance(val, str)
                                else f" DEFAULT {val}"
                            )
                        conn.execute(_sa.text(
                            f"ALTER TABLE {table.name} ADD COLUMN "
                            f"{col.name} {col_type}{default_clause}"
                        ))
                        conn.commit()
                        print(f"[DB Migration] Added column '{col.name}' to '{table.name}'")
            except Exception as migration_err:
                print(f"[DB Migration] Warning for '{table.name}': {migration_err}")


def init_db():
    """Initializes SQLite database and tables, then runs auto-migration."""
    Base.metadata.create_all(bind=engine)
    _migrate_tables()  # Safely add any new columns to existing tables

    # Seed default student demo account if not exists
    session = SessionLocal()
    try:
        demo = session.query(User).filter(User.username == "student").first()
        if not demo:
            demo_user = User(
                username="student",
                full_name="Student Learner",
                email="student@university.edu",
                password_hash=hash_password("student123"),
                user_type="Student / Secondary Income",
                created_at=datetime.utcnow()
            )
            session.add(demo_user)
            session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def register_user(username: str, full_name: str, email: str, password: str, user_type: str = "Student / Secondary Income") -> Tuple[bool, str]:
    """Registers a new user account."""
    init_db()
    session = SessionLocal()
    try:
        clean_user = username.strip().lower()
        if not clean_user or len(clean_user) < 3:
            return False, "Username must be at least 3 characters."
        if not password or len(password) < 4:
            return False, "Password must be at least 4 characters."

        # Check existing
        existing = session.query(User).filter((User.username == clean_user) | (User.email == email.strip().lower())).first()
        if existing:
            return False, "Username or email is already registered. Please sign in."

        new_user = User(
            username=clean_user,
            full_name=full_name.strip() or clean_user.title(),
            email=email.strip().lower(),
            password_hash=hash_password(password),
            user_type=user_type,
            created_at=datetime.utcnow()
        )
        session.add(new_user)
        session.commit()
        return True, "Account created successfully! You can now log in."
    except Exception as e:
        session.rollback()
        return False, f"Registration error: {e}"
    finally:
        session.close()


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticates username and password against database."""
    init_db()
    session = SessionLocal()
    try:
        clean_user = username.strip().lower()
        user = session.query(User).filter(User.username == clean_user).first()
        if not user:
            return None
        
        pwd_hash = hash_password(password)
        if user.password_hash == pwd_hash:
            return {
                "id": user.id,
                "username": user.username,
                "full_name": user.full_name,
                "email": user.email,
                "user_type": user.user_type,
                "created_at": user.created_at.strftime("%Y-%m-%d")
            }
        return None
    finally:
        session.close()


def save_portfolio(
    total_capital: float,
    currency: str,
    risk_profile: str,
    horizon_years: int,
    portfolio_data: Dict[str, Any],
    sim_data: Dict[str, Any],
    sentiment_data: Optional[Dict[str, Any]] = None,
    client_name: str = "Individual Investor",
    username: str = "guest"
) -> int:
    """
    Persists an executed portfolio optimization and scenario run linked to a username.
    """
    init_db()
    session = SessionLocal()
    try:
        sent_score = sentiment_data.get("composite_score", 0.0) if sentiment_data else 0.0
        sent_label = sentiment_data.get("overall_label", "Neutral") if sentiment_data else "Neutral"

        holdings_dict = {
            "stocks": portfolio_data.get("stocks_picks", []),
            "mutual_funds": portfolio_data.get("mfs_picks", []),
            "sip": portfolio_data.get("sip_info", {}),
            "cash_buffer": portfolio_data.get("cash_buffer", 0.0)
        }

        record = PortfolioHistory(
            timestamp=datetime.now(),
            username=username.strip().lower() if username else "guest",
            client_name=client_name,
            currency=currency,
            total_capital=float(total_capital),
            risk_profile=risk_profile,
            horizon_years=int(horizon_years),
            expected_cagr=float(portfolio_data.get("expected_cagr", 14.0)),
            expected_volatility=float(portfolio_data.get("expected_volatility", 15.0)),
            sharpe_ratio=float(portfolio_data.get("sharpe_ratio", 1.0)),
            sentiment_score=float(sent_score),
            sentiment_label=sent_label,
            worst_case_wealth=float(sim_data.get("final_worst", 0.0)),
            average_case_wealth=float(sim_data.get("final_avg", 0.0)),
            best_case_wealth=float(sim_data.get("final_best", 0.0)),
            var_98_pct=float(sim_data.get("var_1y_98_pct", 0.0)),
            holdings_json=json.dumps(holdings_dict)
        )

        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    finally:
        session.close()


def get_all_portfolios(username: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
    """
    Fetches saved portfolio runs filtered by username (or all if not specified).
    """
    init_db()
    session = SessionLocal()
    try:
        query = session.query(PortfolioHistory)
        if username and username != "admin":
            query = query.filter(PortfolioHistory.username == username.strip().lower())
        
        records = query.order_by(desc(PortfolioHistory.timestamp)).limit(limit).all()
        if not records:
            return pd.DataFrame()

        data = []
        for r in records:
            data.append({
                "ID": r.id,
                "Saved At": r.timestamp.strftime("%Y-%m-%d %H:%M"),
                "User": r.username,
                "Portfolio": r.client_name,
                "Currency": r.currency,
                "Capital": r.total_capital,
                "Strategy": r.risk_profile,
                "Horizon": f"{r.horizon_years} Yrs",
                "CAGR": f"{r.expected_cagr:.1f}%",
                "Volatility": f"{r.expected_volatility:.1f}%",
                "Sharpe": f"{r.sharpe_ratio:.2f}",
                "Sentiment": f"{r.sentiment_label} ({r.sentiment_score:+.2f})",
                "Avg Wealth": r.average_case_wealth,
                "Worst Case": r.worst_case_wealth,
                "Best Case": r.best_case_wealth,
                "VaR 98%": f"{r.var_98_pct:.1f}%"
            })
        return pd.DataFrame(data)
    finally:
        session.close()


def delete_portfolio(record_id: int, username: Optional[str] = None) -> bool:
    """Deletes a portfolio record by ID with optional username authorization."""
    init_db()
    session = SessionLocal()
    try:
        query = session.query(PortfolioHistory).filter(PortfolioHistory.id == record_id)
        if username and username != "admin":
            query = query.filter(PortfolioHistory.username == username.strip().lower())
        
        rec = query.first()
        if rec:
            session.delete(rec)
            session.commit()
            return True
        return False
    finally:
        session.close()


# ----------------- PAPER TRADING DESK PERSISTENCE -----------------

def get_or_create_paper_account(username: str, currency: str = "INR", initial_cash: float = 100000.0) -> Dict[str, Any]:
    """Retrieves or initializes a paper trading cash account for the user."""
    init_db()
    session = SessionLocal()
    clean_user = username.strip().lower()
    try:
        acc = session.query(PaperAccount).filter(PaperAccount.username == clean_user).first()
        if not acc:
            acc = PaperAccount(
                username=clean_user,
                currency=currency,
                cash_balance=float(initial_cash),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(acc)
            session.commit()
            session.refresh(acc)
        return {
            "id": acc.id,
            "username": acc.username,
            "currency": acc.currency,
            "cash_balance": acc.cash_balance,
            "updated_at": acc.updated_at.strftime("%Y-%m-%d %H:%M")
        }
    finally:
        session.close()


def execute_paper_order(
    username: str,
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    order_type: str = "MARKET",
    stop_loss: float = 0.0,
    target_price: float = 0.0,
    evidence_notes: str = ""
) -> Tuple[bool, str]:
    """
    Executes a simulated paper order. Validates cash balance for BUYs.
    """
    init_db()
    session = SessionLocal()
    clean_user = username.strip().lower()
    try:
        acc = session.query(PaperAccount).filter(PaperAccount.username == clean_user).first()
        if not acc:
            acc = PaperAccount(username=clean_user, cash_balance=100000.0)
            session.add(acc)
            session.commit()
            session.refresh(acc)

        cost = quantity * price
        side_clean = side.strip().upper()

        if side_clean == "BUY":
            if acc.cash_balance < cost:
                return False, f"Insufficient paper cash ({acc.currency} {acc.cash_balance:,.2f}) for order cost ({acc.currency} {cost:,.2f})."
            acc.cash_balance -= cost

        trade = PaperTrade(
            timestamp=datetime.utcnow(),
            username=clean_user,
            symbol=symbol.strip().upper(),
            side=side_clean,
            order_type=order_type,
            quantity=int(quantity),
            entry_price=float(price),
            stop_loss=float(stop_loss),
            target_price=float(target_price),
            status="OPEN",
            evidence_notes=evidence_notes,
            model_version="QuantNexa Multi-Agent v2.0"
        )
        session.add(trade)
        acc.updated_at = datetime.utcnow()
        session.commit()
        return True, f"Simulated {side_clean} order filled: {quantity} shares of {symbol} at {acc.currency} {price:,.2f}."
    except Exception as e:
        session.rollback()
        return False, f"Paper execution error: {str(e)}"
    finally:
        session.close()


def get_paper_open_positions(username: str) -> pd.DataFrame:
    """Retrieves all active open paper positions for a user."""
    init_db()
    session = SessionLocal()
    clean_user = username.strip().lower()
    try:
        trades = session.query(PaperTrade).filter(
            PaperTrade.username == clean_user,
            PaperTrade.status == "OPEN"
        ).order_by(desc(PaperTrade.timestamp)).all()

        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            rows.append({
                "ID": t.id,
                "Date": t.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Symbol": t.symbol,
                "Side": t.side,
                "Shares": t.quantity,
                "Entry Price": t.entry_price,
                "Invested": round(t.quantity * t.entry_price, 2),
                "Stop-Loss": t.stop_loss,
                "Target": t.target_price,
                "Evidence": t.evidence_notes[:50] + "..." if len(t.evidence_notes) > 50 else t.evidence_notes
            })
        return pd.DataFrame(rows)
    finally:
        session.close()


def close_paper_position(trade_id: int, current_market_price: float, username: str) -> Tuple[bool, str]:
    """Closes an open position, realizes P&L, and returns funds to paper balance."""
    init_db()
    session = SessionLocal()
    clean_user = username.strip().lower()
    try:
        trade = session.query(PaperTrade).filter(
            PaperTrade.id == int(trade_id),
            PaperTrade.username == clean_user,
            PaperTrade.status == "OPEN"
        ).first()

        if not trade:
            return False, "Active paper trade record not found."

        acc = session.query(PaperAccount).filter(PaperAccount.username == clean_user).first()
        exit_val = trade.quantity * current_market_price
        pnl = exit_val - (trade.quantity * trade.entry_price) if trade.side == "BUY" else (trade.quantity * trade.entry_price) - exit_val

        trade.status = "CLOSED"
        trade.exit_price = float(current_market_price)
        trade.exit_timestamp = datetime.utcnow()
        trade.realized_pnl = round(pnl, 2)

        if acc:
            acc.cash_balance += exit_val
            acc.updated_at = datetime.utcnow()

        session.commit()
        return True, f"Closed position on {trade.symbol}. Realized P&L: {pnl:+,.2f}."
    except Exception as e:
        session.rollback()
        return False, f"Failed to close position: {str(e)}"
    finally:
        session.close()


def get_paper_history(username: str, limit: int = 50) -> pd.DataFrame:
    """Retrieves full paper trade log."""
    init_db()
    session = SessionLocal()
    clean_user = username.strip().lower()
    try:
        trades = session.query(PaperTrade).filter(
            PaperTrade.username == clean_user
        ).order_by(desc(PaperTrade.timestamp)).limit(limit).all()

        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            rows.append({
                "ID": t.id,
                "Timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Symbol": t.symbol,
                "Side": t.side,
                "Qty": t.quantity,
                "Entry": t.entry_price,
                "Exit": t.exit_price if t.exit_price else "-",
                "Status": t.status,
                "Realized P&L": t.realized_pnl if t.status == "CLOSED" else "-",
                "Notes": t.evidence_notes
            })
        return pd.DataFrame(rows)
    finally:
        session.close()
