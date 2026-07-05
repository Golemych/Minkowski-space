import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from arena import ArenaClient
from indicators import Indicators

BASE_DIR = Path(__file__).parent
PAIRS_PATH = BASE_DIR / "pairs.json"
CACHE_DIR = BASE_DIR / "cache"

TIMEFRAME = "15m"
BARS_DAYS = 7
BOLLINGER_LENGTH = 230
BOLLINGER_DEVIATION = 2.1
KELTNER_EMA_LENGTH = 150
KELTNER_ATR_LENGTH = 24
KELTNER_DEVIATION = 3.9
DECIMALS_MAP = {
    "SBER": 2, "SBERP": 2, "GAZP": 2, "ROSN": 2,
    "LKOH": 2, "VTBR": 2, "GMKN": 2, "ALRS": 2,
    "AFLT": 2, "MGNT": 2,
}


def _load_pairs() -> dict:
    if not PAIRS_PATH.exists():
        return {}
    with open(PAIRS_PATH) as f:
        return json.load(f)


def _save_pairs(data: dict):
    with open(PAIRS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _get_client() -> ArenaClient:
    load_dotenv(BASE_DIR / ".env")
    token = os.environ.get("ARENA_API_TOKEN", "")
    aid = int(os.environ.get("ARENA_ACCOUNT_ID", "0"))
    return ArenaClient(token, aid, cache_dir=CACHE_DIR)


def _run_async(coro):
    return asyncio.run(coro)


def _calc_contango_coeff(ticker: str) -> float:
    ticker_upper = ticker.upper()
    now = datetime.now()
    if "VTB" in ticker_upper or "VTBR" in ticker_upper:
        if now.year < 2024: return 20
        if now.year == 2024 and (now.month < 7 or (now.month == 7 and now.day < 15)): return 20
        return 100
    if "GMKN" in ticker_upper:
        if now.year < 2024: return 100
        if now.year == 2024 and (now.month < 4 or (now.month == 4 and now.day < 4)): return 100
        return 10
    decimals = DECIMALS_MAP.get(ticker_upper, 2)
    return 10 ** decimals


st.set_page_config(page_title="Futures Bot", layout="wide")
st.title("Futures Bot")

pairs = _load_pairs()
tab = st.sidebar.radio("Section", ["Dashboard", "Charts", "Pairs", "Trades", "Safety"])

if tab == "Dashboard":
    st.header("Account & Status")
    client = _get_client()

    bot_pid_path = BASE_DIR / ".bot.pid"
    is_running = bot_pid_path.exists()
    try:
        if is_running:
            pid = int(bot_pid_path.read_text().strip())
            proc = subprocess.run(f"tasklist /FI \"PID eq {pid}\" /NH", capture_output=True, text=True, shell=True)
            is_running = str(pid) in proc.stdout
            if not is_running:
                bot_pid_path.unlink(missing_ok=True)
    except Exception:
        is_running = False

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if is_running:
            st.success(f"Bot is running (PID {bot_pid_path.read_text().strip()})")
            if st.button("Stop bot"):
                try:
                    pid = int(bot_pid_path.read_text().strip())
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                    bot_pid_path.unlink(missing_ok=True)
                    st.rerun()
                except Exception:
                    st.error("Failed to stop")
        else:
            bot_pid_path.unlink(missing_ok=True)
            if st.button("Start bot"):
                proc = subprocess.Popen(
                    [sys.executable, "-u", "main.py"],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                bot_pid_path.write_text(str(proc.pid))
                st.rerun()

    with col_b:
        trades_path = BASE_DIR / "trades.csv"
        if trades_path.exists():
            df = pd.read_csv(trades_path)
            st.metric("Trades", len(df))
        else:
            st.metric("Trades", "0")

    with st.spinner("Loading account..."):
        try:
            acc = _run_async(client.get_account())
            equity = float(acc.get("equity", {}).get("value", 0))
            cash = float(acc.get("cash", {}).get("value", 0))
            avail = float(acc.get("available_cash", {}).get("value", 0))
            positions = acc.get("positions", [])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Equity", f"{equity:,.0f}")
            col2.metric("Cash", f"{cash:,.0f}")
            col3.metric("Available", f"{avail:,.0f}")
            col4.metric("Open positions", len(positions))

            if positions:
                st.subheader("Open positions")
                for p in positions:
                    sym = p.get("symbol", "?")
                    qty = float(p.get("quantity", {}).get("value", 0))
                    side = "BUY" if qty > 0 else "SELL"
                    qty_abs = abs(int(qty))
                    avg = float(p.get("average_price", {}).get("value", 0))
                    pnl = float(p.get("unrealized_pnl", {}).get("value", 0))

                    c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1.5, 1, 1.5])
                    c1.write(f"**{sym}**")
                    c2.write(side)
                    c3.write(f"{qty_abs}")
                    c4.write(f"{avg:.2f}")
                    c5.write(f"{pnl:+.2f}")
                    if c6.button("Sell", key=f"close_{sym}"):
                        close_side = "SIDE_SELL" if side == "BUY" else "SIDE_BUY"
                        _run_async(client.place_order(sym, close_side, qty_abs))
                        st.success(f"Close order sent for {sym}")
                        st.rerun()
        except Exception as e:
            st.error(f"Account load failed: {e}")

elif tab == "Charts":
    st.header("Charts")

    if not pairs:
        st.warning("No pairs configured in pairs.json")
        st.stop()

    ticker = st.selectbox("Instrument", list(pairs.keys()))
    pair = pairs[ticker]
    strategy = pair.get("strategy", "bollinger")
    spot_sym = pair.get("spot", ticker)
    futures_sym = pair.get("futures", "")

    source = st.radio("Source", ["Spot", "Futures"], horizontal=True)
    sym = futures_sym if source == "Futures" else spot_sym

    client = _get_client()
    with st.spinner(f"Loading {sym} ..."):
        df = _run_async(client.get_bars(sym, TIMEFRAME, BARS_DAYS))

    if df is None or df.empty:
        st.error(f"No data for {sym}")
        st.stop()

    df.columns = [c.lower() for c in df.columns]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=sym,
    ))

    if strategy == "keltner":
        k_u, k_m, k_l = Indicators.keltner_channel(df, KELTNER_EMA_LENGTH, KELTNER_ATR_LENGTH, KELTNER_DEVIATION)
        fig.add_trace(go.Scatter(x=df.index, y=k_u, mode="lines",
                                 name="KC Upper", line=dict(color="orange", dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=k_m, mode="lines",
                                 name="KC Mid", line=dict(color="orange", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=k_l, mode="lines",
                                 name="KC Lower", line=dict(color="orange", dash="dot")))
        strategy_label = "Keltner"
    else:
        bb_u, bb_m, bb_l = Indicators.bollinger_bands(df, BOLLINGER_LENGTH, BOLLINGER_DEVIATION)
        fig.add_trace(go.Scatter(x=df.index, y=bb_u, mode="lines",
                                 name="BB Upper", line=dict(color="blue", dash="dash")))
        fig.add_trace(go.Scatter(x=df.index, y=bb_m, mode="lines",
                                 name="BB Mid", line=dict(color="blue", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=bb_l, mode="lines",
                                 name="BB Lower", line=dict(color="blue", dash="dash")))
        strategy_label = "Bollinger"

    fig.update_layout(template="plotly_dark", height=600,
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{df['close'].iloc[-1]:.2f}")
    c2.metric("Strategy", strategy_label)
    coeff = _calc_contango_coeff(ticker)
    c3.metric("Contango coeff", coeff)
    c4.metric("Bars", len(df))

elif tab == "Pairs":
    st.header("Futures pairs")

    to_delete = []
    for ticker, data in list(pairs.items()):
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1.5, 1.5, 1, 1, 1])
        c1.write(f"**{ticker}**")
        data["spot"] = c2.text_input("Spot", data.get("spot", ""), key=f"sp_{ticker}")
        data["futures"] = c3.text_input("Futures", data.get("futures", ""), key=f"fu_{ticker}")
        opts = {"bollinger": "Bollinger", "keltner": "Keltner"}
        current = data.get("strategy", "bollinger")
        data["strategy"] = c4.selectbox("Strategy", list(opts.keys()),
                                         format_func=lambda x: opts[x],
                                         index=list(opts.keys()).index(current) if current in opts else 0,
                                         key=f"st_{ticker}")
        coeff = _calc_contango_coeff(ticker)
        c5.metric("Coeff", f"{coeff:.0f}")
        data["enabled"] = c6.checkbox("On", data.get("enabled", True), key=f"en_{ticker}")
        if st.button("Delete", key=f"del_{ticker}"):
            to_delete.append(ticker)

    for t in to_delete:
        del pairs[t]

    st.divider()
    with st.expander("Add pair"):
        c1, c2, c3, c4 = st.columns([1, 1.5, 1.5, 1])
        nt = c1.text_input("Ticker", placeholder="SBER")
        ns = c2.text_input("Spot", placeholder="SBER@MISX")
        nf = c3.text_input("Futures", placeholder="SBER-9.26@FORTS")
        ns_strat = c4.selectbox("Strategy", ["bollinger", "keltner"])
        if st.button("Add") and nt:
            pairs[nt] = {"spot": ns, "futures": nf, "strategy": ns_strat, "enabled": True}
            _save_pairs(pairs)
            st.success(f"Added {nt}")
            st.rerun()

    if st.button("Save all pairs"):
        _save_pairs(pairs)
        st.success(f"Saved {len(pairs)} pairs")

elif tab == "Trades":
    st.header("Trades log")
    path = BASE_DIR / "trades.csv"
    if path.exists():
        df = pd.read_csv(path)
        st.metric("Total trades", len(df))
        st.dataframe(df.tail(200)[::-1], use_container_width=True)
        if "pnl" in df.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total P&L", f"{df['pnl'].sum():.2f}")
            nonzero = df[df["pnl"] != 0]
            if len(nonzero):
                c2.metric("Winrate", f"{(nonzero['pnl']>0).mean()*100:.1f}%")
                c3.metric("Avg P&L", f"{nonzero['pnl'].mean():.2f}")
    else:
        st.info("No trades yet")

elif tab == "Safety":
    st.header("Emergency position close")
    st.warning("Closes ALL open positions at market!")

    client = _get_client()
    positions = _run_async(client.get_positions())

    if positions:
        st.dataframe(pd.DataFrame(positions))
        if st.button("CLOSE ALL", type="primary"):
            if st.button("Confirm CLOSE ALL"):
                for pos in positions:
                    side = "SIDE_SELL" if pos["side"] == "BUY" else "SIDE_BUY"
                    _run_async(client.place_order(pos["symbol"], side, int(pos["quantity"])))
                st.success("Close orders sent")
                st.rerun()
    else:
        st.info("No open positions")
