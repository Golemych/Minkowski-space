"""
Streamlit UI for Minkowski bot v2.

This UI starts/stops the v2 bot process and reads state.json/trades.csv
written by MetricsCollector.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from arena_client import ArenaClient

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
CONFIG_EXAMPLE_PATH = BASE_DIR / "config.example.json"
STATE_PATH = BASE_DIR / "state.json"
TRADES_PATH = BASE_DIR / "trades.csv"
STOP_FLAG = BASE_DIR / "stop.flag"
PID_PATH = BASE_DIR / ".bot.pid"
LOG_PATH = BASE_DIR / ".bot.log"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

os.chdir(BASE_DIR)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_config() -> dict:
    config = _load_json(CONFIG_PATH)
    if config:
        return config
    return _load_json(CONFIG_EXAMPLE_PATH)


def save_config(config: dict):
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_state() -> dict:
    return _load_json(STATE_PATH)


def is_bot_running() -> bool:
    if not PID_PATH.exists():
        return False

    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
    except ValueError:
        PID_PATH.unlink(missing_ok=True)
        return False

    result = subprocess.run(
        f'tasklist /FI "PID eq {pid}" /NH',
        capture_output=True,
        text=True,
        shell=True,
    )
    running = str(pid) in result.stdout
    if not running:
        PID_PATH.unlink(missing_ok=True)
    return running


def start_bot():
    if is_bot_running():
        return

    log_file = open(LOG_PATH, "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        cwd=str(BASE_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW,
    )
    PID_PATH.write_text(str(process.pid), encoding="utf-8")
    time.sleep(1)

    if process.poll() is not None:
        log_file.close()
        PID_PATH.unlink(missing_ok=True)
        raise RuntimeError(LOG_PATH.read_text(encoding="utf-8", errors="replace"))

    log_file.close()


def stop_bot():
    STOP_FLAG.write_text("1", encoding="utf-8")


def format_money(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "0.00"


def get_arena_client() -> ArenaClient:
    if CONFIG_PATH.exists():
        return ArenaClient(str(CONFIG_PATH))
    if CONFIG_EXAMPLE_PATH.exists():
        return ArenaClient(str(CONFIG_EXAMPLE_PATH))
    raise FileNotFoundError("No config.json or config.example.json found")


def read_log_tail(limit: int = 80) -> str:
    if not LOG_PATH.exists():
        return ""
    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


st.set_page_config(page_title="Minkowski v2", layout="wide")
st.title("Minkowski v2")
st.caption("Lorentzian Classification + production execution architecture")

config = load_config()
state = load_state()
running = is_bot_running()

tab = st.sidebar.radio(
    "Раздел",
    ["Dashboard", "Positions", "Trades", "Settings", "Logs"],
)

with st.sidebar:
    st.divider()
    if running:
        st.success(f"Bot is running, PID {PID_PATH.read_text().strip()}")
        if st.button("Stop bot", width="stretch"):
            stop_bot()
            st.info("Stop flag sent")
            time.sleep(1)
            st.rerun()
    else:
        st.warning("Bot is stopped")
        if st.button("Start bot", width="stretch"):
            try:
                if not CONFIG_PATH.exists() and CONFIG_EXAMPLE_PATH.exists():
                    save_config(config)
                start_bot()
                st.success("Bot started")
                st.rerun()
            except Exception as exc:
                st.error("Bot failed to start")
                st.code(str(exc), language="text")


if tab == "Dashboard":
    status = state.get("status", "IDLE")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status", status)
    col2.metric("Equity", format_money(state.get("equity", 0)))
    col3.metric("Cash", format_money(state.get("cash", 0)))
    col4.metric("Positions", state.get("open_positions", 0))
    col5.metric("Daily orders", state.get("daily_orders", 0))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Drawdown", f"{float(state.get('drawdown_pct', 0)) * 100:.2f}%")
    col_b.metric("Stocks", state.get("stocks_count", len(config.get("stocks", []))))
    col_c.metric("Last update", str(state.get("last_update", "-"))[:19])

    st.subheader("Last signals")
    for signal in state.get("last_signals", ["No signals"])[-20:]:
        st.write(signal)

    if state.get("last_errors"):
        st.subheader("Errors")
        for error in state["last_errors"]:
            st.warning(error)

    st.subheader("Decision stats")
    stats = state.get("stats", {})
    if stats:
        st.json(stats)
    else:
        st.info("No stats yet")

    time.sleep(3)
    st.rerun()


elif tab == "Positions":
    positions = state.get("positions", [])
    if positions:
        for p in positions:
            sym = p.get("symbol", "?")
            side = p.get("side", "?")
            qty = p.get("quantity", 0)
            entry = p.get("entry_price", 0)
            current = p.get("current_price", 0)
            pnl = p.get("pnl_pct", 0)
            strat = p.get("strategy", "MK")
            bars = p.get("bars_held", 0)
            sl = p.get("stop_loss", 0)
            tp = p.get("take_profit", 0)
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([2, 1, 1, 1.5, 1, 1, 1, 1.5])
            c1.write(f"**{sym}**")
            c2.write(side)
            c3.write(f"{qty:.0f}")
            c4.write(f"${entry:.2f}")
            c5.write(f"{pnl:+.1f}%")
            c6.write(strat)
            c7.write(f"{bars}b")
            if c8.button("Sell", key=f"close_{sym}"):
                try:
                    client = get_arena_client()
                    close_side = "SIDE_SELL" if side == "BUY" else "SIDE_BUY"
                    result = client.place_market_order(sym, close_side, int(qty))
                    if result:
                        st.success(f"Close order sent for {sym}")
                    else:
                        st.error(f"Order failed for {sym}")
                except Exception as ex:
                    st.error(f"Error: {ex}")
                st.rerun()
    else:
        st.info("No open positions in state.json")


elif tab == "Trades":
    if TRADES_PATH.exists():
        trades = pd.read_csv(TRADES_PATH)
        st.dataframe(trades.tail(200), width="stretch")
    else:
        st.info("No trades.csv yet")


elif tab == "Settings":
    st.subheader("Core settings")
    edited = dict(config)

    c1, c2, c3 = st.columns(3)
    edited["timeframe"] = c1.text_input("Timeframe", edited.get("timeframe", "15m"))
    edited["bars_depth_days"] = c2.number_input(
        "Bars depth days",
        min_value=1,
        value=int(edited.get("bars_depth_days", 30)),
    )
    edited["neighbors_count"] = c3.number_input(
        "Neighbors",
        min_value=1,
        value=int(edited.get("neighbors_count", 8)),
    )

    c4, c5, c6 = st.columns(3)
    edited["min_confidence"] = c4.number_input(
        "Min confidence",
        min_value=0.0,
        max_value=1.0,
        value=float(edited.get("min_confidence", 0.25)),
        step=0.05,
    )
    edited["cooldown_bars"] = c5.number_input(
        "Cooldown bars",
        min_value=0,
        value=int(edited.get("cooldown_bars", 2)),
    )
    edited["max_open_positions"] = c6.number_input(
        "Max positions",
        min_value=0,
        value=int(edited.get("max_open_positions", 5)),
    )

    st.subheader("Stocks")
    stocks_text = st.text_area(
        "One symbol per line",
        "\n".join(edited.get("stocks", [])),
        height=220,
    )
    edited["stocks"] = [
        line.strip()
        for line in stocks_text.splitlines()
        if line.strip()
    ]

    if st.button("Save settings", width="stretch"):
        save_config(edited)
        st.success("Saved to config.json")


elif tab == "Logs":
    st.subheader("Bot log")
    log_text = read_log_tail()
    if log_text:
        st.code(log_text, language="text")
    else:
        st.info("No .bot.log yet")

