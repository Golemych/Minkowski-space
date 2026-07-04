import asyncio
import streamlit as st
import sys
import subprocess
from pathlib import Path
import os
from datetime import datetime  # Correct import
import threading
import logging
from dataclasses import dataclass
import pandas as pd
import plotly.graph_objects as go
import json
from arena_client import ArenaClient
from indicators import KernelRegression, Filters, normalize_df
from strategy import StrategyManager, MinkowskiClassifier

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
TRADES_PATH = BASE_DIR / "trades.csv"
STATE_PATH = BASE_DIR / "state.json"
STRATEGY_NAME = "MK"

@dataclass
class State:
    status: str = "IDLE"
    balance: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    available_cash: float = 0.0
    open_positions: int = 0
    daily_orders: int = 0
    drawdown_pct: float = 0.0
    last_signals: list[str] = None
    last_errors: list[str] = None
    stats: dict[str, int] = None
    positions_owners: dict[str, str] = None
    last_update: str = None

def _get_client() -> ArenaClient:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    token = os.environ.get("ARENA_API_TOKEN", "")
    aid = int(os.environ.get("ARENA_ACCOUNT_ID", "0"))
    return ArenaClient(token, aid, cache_dir=BASE_DIR / "cache")

def _run_async(coro):
    return asyncio.run(coro)

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(cfg: dict):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def load_state() -> State:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                return State(**json.load(f))
        except Exception:
            pass
    return State()

def save_state(state: State):
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state.__dict__, f, indent=2, ensure_ascii=False)

def format_rub(value) -> str:
    return f"{value:,.0f} ₽"

# ── Bot control via PID file ───────────────────────────────────────
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

cfg = load_config()
state = load_state()
if not state:
    state = State()

# Load live account if bot not running
if not is_running and not state.status in ("RUNNING",):
    try:
        client = _get_client()
        acc = _run_async(client.get_account())
        pos = _run_async(client.get_positions())
        state.update({
            "status": "IDLE",
            "balance": float(acc.get("available_cash", {}).get("value", 0)),
            "equity": float(acc.get("equity", {}).get("value", 0)),
            "cash": float(acc.get("cash", {}).get("value", 0)),
            "available_cash": float(acc.get("available_cash", {}).get("value", 0)),
            "open_positions": len(pos),
            "last_update": datetime.now().isoformat(),
        })
    except Exception as e:
        st.error(f"API error: {e}")

# ── Streamlit UI Layout ────────────────────────────────────────
tab = st.sidebar.radio("Раздел", [
    "📊 Dashboard",
    "⚙️ Настройки ML",
    "📈 Акции",
    "📋 Сделки",
    "📉 Графики",
    "🚨 Безопасность"
])

# ============================================================
# TAB: DASHBOARD
# ============================================================
if tab == "📊 Dashboard":
    st.header("Текущее состояние бота")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💎 Капитал", format_rub(state.equity or state.balance))
    col2.metric("💰 Свободно", format_rub(state.cash))
    col3.metric("✅ Доступно", format_rub(state.available_cash))
    col4.metric("📊 Позиций", state.open_positions)
    col5.metric("🎯 Заявок", f"{state.daily_orders}/{cfg.get('max_daily_orders', 190)}")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("📉 Просадка", f"{state.drawdown_pct * 100:.2f}%")
    status_emoji = {
        'RUNNING': '🟢',
        'IDLE': '⏸',
        'STOPPED': '🛑',
        'ERROR': '❌',
        'LIMIT_REACHED': '⚠️'
    }.get(state.status, '❓')
    col_b.metric("🚦 Статус", f"{status_emoji} {state.status}")
    col_c.metric("🕐 Обновление", str(state.last_update)[:19])
    
    # Bot control
    st.subheader("Управление")
    c1, c2 = st.columns(2)
    with c1:
        if is_running:
            st.success(f"Бот запущен (PID {bot_pid_path.read_text().strip()})")
            if st.button("⏹ Остановить бота"):
                try:
                    pid = int(bot_pid_path.read_text().strip())
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                    bot_pid_path.unlink(missing_ok=True)
                    st.rerun()
                except Exception:
                    st.error("Не удалось остановить")
        else:
            bot_pid_path.unlink(missing_ok=True)
            if st.button("▶️ Запустить бота"):
                proc = subprocess.Popen(
                    [sys.executable, "-u", "main.py"],
                    cwd=str(BASE_DIR),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                bot_pid_path.write_text(str(proc.pid))
                st.rerun()
    with c2:
        if st.button("⏹ Остановить бота (флаг)"):
            with open(BASE_DIR / "stop.flag", "w") as f:
                f.write("1")
            st.info("Флаг остановки создан")

# ============================================================
# TAB: ML SETTINGS
# ============================================================
elif tab == "⚙️ Настройки ML":
    st.header("Настройка параметров стратегии MK")
    
    # Load config
    cfg = load_config()
    if not cfg:
        st.error("❌ Ошибка загрузки конфигурации. Проверьте config.json.")
        st.stop()
    
    with st.form(key='ml_settings'):
        drawdown_threshold = st.number_input(
            "Порог просадки (%)", min_value=0.0, max_value=100.0, value=cfg.get('drawdown_threshold', 5.0) * 100, step=0.1
        )
        bars_depth_days = st.number_input(
            "Глубина данных (дней)", min_value=1, max_value=90, value=cfg.get('bars_depth_days', 30), step=1
        )
        max_daily_orders = st.number_input(
            "Максимальное кол-во заявок в день", min_value=1, max_value=500, value=cfg.get('max_daily_orders', 190), step=1
        )
        
        stocks = cfg.get('stocks', [])
        st.write("Акции для торговли:")
        for i in range(len(stocks)):
            symbol = stocks[i].get('symbol', '')
            side = stocks[i].get('side', 'BUY')
            quantity = stocks[i].get('quantity', 1)
            
            st.write(f"Акция {i+1}")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                symbol_input = st.text_input(f"Тикер", value=symbol, key=f'symbol_{i}')
            with col_b:
                side_input = st.selectbox(f"Сторона", options=['BUY', 'SELL'], index=0 if side == 'BUY' else 1, key=f'side_{i}')
            with col_c:
                quantity_input = st.number_input(f"Количество", min_value=1, max_value=1000, value=quantity, step=1, key=f'quantity_{i}')
            
        if st.form_submit_button('Сохранить настройки'):
            cfg['drawdown_threshold'] = drawdown_threshold / 100.0
            cfg['bars_depth_days'] = bars_depth_days
            cfg['max_daily_orders'] = max_daily_orders
            cfg['stocks'] = [
                {"symbol": symbol_input, "side": side_input, "quantity": quantity_input}
                for symbol_input, side_input, quantity_input in zip(
                    [st.session_state[f'symbol_{i}'] for i in range(len(stocks))],
                    [st.session_state[f'side_{i}'] for i in range(len(stocks))],
                    [st.session_state[f'quantity_{i}'] for i in range(len(stocks))]
                )
            ]
            save_config(cfg)
            st.success("Настройки сохранены")

# ============================================================
# TAB: STOCKS
# ============================================================
elif tab == "📈 Акции":
    st.header("Акции")
    
    # Load config
    cfg = load_config()
    if not cfg:
        st.error("❌ Ошибка загрузки конфигурации. Проверьте config.json.")
        st.stop()
    
    stocks = cfg.get('stocks', [])
    for stock in stocks:
        symbol = stock.get('symbol', '')
        side = stock.get('side', 'BUY')
        quantity = stock.get('quantity', 1)
        
        st.write(f"Акция: {symbol}")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.text_input(f"Тикер", value=symbol, disabled=True)
        with col_b:
            st.selectbox(f"Сторона", options=['BUY', 'SELL'], index=0 if side == 'BUY' else 1, disabled=True)
        with col_c:
            st.number_input(f"Количество", min_value=1, max_value=1000, value=quantity, step=1, disabled=True)

# ============================================================
# TAB: TRADES
# ============================================================
elif tab == "📋 Сделки":
    st.header("Сделки")
    
    # Load trades
    if os.path.exists(TRADES_PATH):
        try:
            trades = pd.read_csv(TRADES_PATH)
        except Exception as e:
            st.error(f"❌ Ошибка загрузки сделок: {e}")
            trades = pd.DataFrame()
    else:
        trades = pd.DataFrame()
    
    if not trades.empty:
        st.dataframe(trades, width='stretch')
    else:
        st.info("Сделок пока нет")

# ============================================================
# TAB: CHARTS
# ============================================================
elif tab == "📉 Графики":
    st.header("Графики")
    
    # Load config
    cfg = load_config()
    if not cfg:
        st.error("❌ Ошибка загрузки конфигурации. Проверьте config.json.")
        st.stop()
    
    stocks = cfg.get('stocks', [])
    selected = st.selectbox("Выберите акцию", [s["symbol"] for s in stocks])
    timeframe = st.select_slider("Интервал времени", options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], value="15m")
    days = min(cfg.get('bars_depth_days', 30), 30)
    
    with st.spinner(f"📥 Загрузка {selected} через MOEX ISS..."):
        try:
            client = _get_client()
            df = _run_async(client.get_bars(selected, timeframe=timeframe, days=int(days)))
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
            st.stop()
    
    if df is None or df.empty:
        st.error(f"❌ Не удалось загрузить данные для {selected}")
        st.info("Возможно, рынок закрыт или инструмент не торгуется.")
        st.stop()
    
    try:
        df = normalize_df(df)
    except Exception as e:
        st.error(f"❌ Ошибка нормализации: {e}")
        st.stop()
    
    # Plot charts
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.update_layout(title=f"График {selected}", xaxis_title="Дата", yaxis_title="Цена")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB: SAFETY
# ============================================================
elif tab == "🚨 Безопасность":
    st.header("🚨 Аварийное управление")
    st.warning("⚠️ Используйте эти кнопки только в экстренных случаях!")
    
    client = _get_client()
    
    st.subheader("📋 Текущие открытые позиции")
    with st.spinner("Загрузка позиций..."):
        positions = _run_async(client.get_positions())
    
    if positions:
        pos_df = pd.DataFrame(positions)
        available_cols = [c for c in ["symbol", "side", "quantity", "avg_price", "unrealized_pnl"] if c in pos_df.columns]
        st.dataframe(pos_df[available_cols], width='stretch')
        
        total_pnl = sum(float(p.get("unrealized_pnl", 0)) for p in positions)
        st.metric("💰 Общий нереализованный P&L", f"{total_pnl:+.2f} ₽")
    else:
        st.info("🎉 Открытых позиций нет")
    
    st.divider()
    
    # Кнопка закрытия всех позиций
    st.subheader("🔥 Закрыть ВСЁ по рынку")
    st.error("⚠️ Мгновенно закроет ВСЕ открытые позиции рыночными ордерами!")
    
    if st.button("🚨 ЗАКРЫТЬ ВСЕ ПОЗИЦИИ", type="primary", width='stretch'):
        if not positions:
            st.warning("Нет позиций для закрытия")
        else:
            st.error(f"⚠️ Подтвердите закрытие {len(positions)} позиций!")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ ДА, ЗАКРЫТЬ ВСЁ", key="confirm_close_all"):
                    with st.spinner("Закрытие позиций..."):
                        result = {"success": 0, "failed": 0, "details": []}
                        for pos in positions:
                            try:
                                sym = pos["symbol"]
                                qty = int(pos["quantity"])
                                side = "SIDE_SELL" if pos["side"] == "BUY" else "SIDE_BUY"
                                res = _run_async(client.place_order(sym, side, qty))
                                
                                if res:
                                    result["success"] += 1
                                    result["details"].append(f"✅ {sym}: закрыто {qty}")
                                else:
                                    result["failed"] += 1
                                    result["details"].append(f"❌ {sym}: ошибка API")
                            except Exception as e:
                                result["failed"] += 1
                                result["details"].append(f"❌ {sym}: {e}")
                        
                        if result["success"] > 0:
                            st.success(f"✅ Закрыто позиций: {result['success']}")
                        if result["failed"] > 0:
                            st.error(f"❌ Ошибок: {result['failed']}")
                        
                        with st.expander("📋 Детали"):
                            for d in result["details"]:
                                st.write(d)
                        
                        # Очищаем positions_state.json
                        try:
                            p = BASE_DIR / "positions_state.json"
                            if p.exists():
                                p.unlink()
                        except Exception:
                            pass
                        
                        st.rerun()
            with c2:
                if st.button("❌ Отмена", key="cancel_close_all"):
                    st.info("Отменено")
    
    st.divider()
    
    # Создание stop.flag
    st.subheader("⛔ Принудительная остановка бота")
    if st.button("🛑 Создать stop.flag", width='stretch'):
        with open(BASE_DIR / "stop.flag", "w") as f:
            f.write("1")
        st.success("✅ stop.flag создан. Бот остановится в течение минуты.")