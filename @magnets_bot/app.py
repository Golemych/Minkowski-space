"""
Streamlit UI для стратегии Минковского (акции MOEX)
Управление ботом, настройка параметров, мониторинг позиций и графиков.

Использует:
- arena_client.py — для данных (MOEX ISS) и ордеров (Arena API)
- bot.py — торговый движок
- strategy.py — математика MK
- indicators.py — Kernel Regression для визуализации
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
import json
import os
import time
import threading
import logging
from datetime import datetime
from arena_client import ArenaClient
from bot import TradingBot, CONFIG_PATH, TRADES_PATH, STATE_PATH, STRATEGY_NAME
from indicators import KernelRegression, Filters, normalize_df
from strategy import StrategyManager, MinkowskiClassifier

os.chdir(Path(__file__).parent)

logging.basicConfig(level=logging.WARNING)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

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


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def format_rub(value) -> str:
    """Форматирование рублей"""
    try:
        return f"{float(value):,.0f} ₽"
    except Exception:
        return "0 ₽"


# ============================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================

st.title("🧠 Стратегия: Пространство Минковского")
st.caption("kNN с метрикой Минковского + Kernel Regression")

cfg = load_config()

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
    
    state = load_state()
    
    # Если бот не запущен — загружаем данные напрямую из API
    if not state or state.get('status') in [None, 'IDLE']:
        st.info("ℹ️ Бот не запущен. Загружаем текущие данные из Arena API...")
        with st.spinner("Загрузка..."):
            try:
                client = ArenaClient(CONFIG_PATH)
                info = client.get_account_info()
                positions = client.get_positions()
                
                if info:
                    state = {
                        "status": "IDLE",
                        "balance": info.get("available_cash", 0),
                        "equity": info.get("equity", 0),
                        "cash": info.get("cash", 0),
                        "available_cash": info.get("available_cash", 0),
                        "open_positions": len(positions),
                        "daily_orders": 0,
                        "drawdown_pct": 0,
                        "last_signals": ["⏸ Бот не запущен"],
                        "last_errors": [],
                        "stats": {"BUY": 0, "SELL": 0, "CLOSE_LONG": 0, "CLOSE_SHORT": 0},
                        "positions_owners": {},
                        "last_update": datetime.now().isoformat(),
                    }
                else:
                    st.error("❌ Не удалось получить данные из API. Проверьте api_secret в настройках.")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
    
    if not state:
        st.stop()
    
    # Метрики
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💎 Капитал", format_rub(state.get('equity', state.get('balance', 0))))
    col2.metric("💰 Свободно", format_rub(state.get('cash', 0)))
    col3.metric("✅ Доступно", format_rub(state.get('available_cash', 0)))
    col4.metric("📊 Позиций", state.get('open_positions', 0))
    col5.metric("🎯 Заявок", f"{state.get('daily_orders', 0)}/{cfg.get('max_daily_orders', 190)}")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("📉 Просадка", f"{state.get('drawdown_pct', 0)*100:.2f}%")
    
    status = state.get('status', 'IDLE')
    status_emoji = {
        'RUNNING': '🟢',
        'IDLE': '⏸',
        'STOPPED': '🛑',
        'ERROR': '❌',
        'LIMIT_REACHED': '⚠️'
    }.get(status, '❓')
    col_b.metric("🚦 Статус", f"{status_emoji} {status}")
    col_c.metric("🕐 Обновление", state.get('last_update', '—')[:19] if state.get('last_update') else '—')
    
    # Управление
    st.subheader("Управление")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Запустить бота", width='stretch'):
            if not cfg.get("api_secret"):
                st.error("❌ Сначала заполните API токен в Настройках!")
            else:
                save_config(cfg)
                def run_bot():
                    bot = TradingBot()
                    bot.run()
                threading.Thread(target=run_bot, daemon=True).start()
                st.success("✅ Бот запущен в фоне")
    with c2:
        if st.button("⏹ Остановить бота", width='stretch'):
            with open("stop.flag", "w") as f:
                f.write("1")
            st.info("⛔ Команда остановки отправлена")
    
    # Статистика действий
    st.subheader("📊 Статистика действий")
    stats = state.get('stats', {})
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("🟢 BUY", stats.get('BUY', 0))
    sc2.metric("🔴 SELL", stats.get('SELL', 0))
    sc3.metric("⬆️ CLOSE LONG", stats.get('CLOSE_LONG', 0))
    sc4.metric("⬇️ CLOSE SHORT", stats.get('CLOSE_SHORT', 0))
    
    # Открытые позиции
    if state.get('positions_owners'):
        st.subheader("📂 Открытые позиции")
        pos_df = pd.DataFrame([
            {"Тикер": k, "Сторона": v.get('side'), "Цена входа": v.get('entry_price'), 
             "Открыта": v.get('opened_at', '')[:19]}
            for k, v in state['positions_owners'].items()
        ])
        st.dataframe(pos_df, width='stretch')
    
    # Последние сигналы
    if state.get("last_signals"):
        st.subheader("📡 Последние сигналы")
        for sig in state["last_signals"][-10:]:
            if "✅" in sig or "BUY" in sig:
                st.success(sig)
            elif "🔻" in sig or "CLOSE" in sig or "SELL" in sig:
                st.warning(sig)
            elif "⛔" in sig or "🛑" in sig:
                st.error(sig)
            else:
                st.info(sig)
    
    # Ошибки
    if state.get("last_errors"):
        st.subheader("⚠️ Ошибки")
        for err in state["last_errors"]:
            st.warning(err)
    
    # Автообновление
    time.sleep(5)
    st.rerun()


# ============================================================
# TAB: НАСТРОЙКИ ML
# ============================================================

elif tab == "⚙️ Настройки ML":
    st.header("Настройки Минковского")
    
    with st.form("settings_form"):
        st.subheader("🔑 Finam Arena API")
        cfg["api_secret"] = st.text_input(
            "API Secret Token", 
            value=cfg.get("api_secret", ""), 
            type="password"
        )
        cfg["account_id"] = st.text_input(
            "Account ID", 
            value=str(cfg.get("account_id", ""))
        )
        cfg["base_url"] = st.text_input(
            "Base URL", 
            value=cfg.get("base_url", "https://arena.finam.ru/v1")
        )
        
        st.subheader("🤖 Machine Learning (MK)")
        c1, c2, c3 = st.columns(3)
        cfg["neighbors_count"] = c1.number_input(
            "Neighbors (k)", 1, 100, 
            cfg.get("neighbors_count", 8),
            help="Количество ближайших соседей"
        )
        cfg["max_bars_back"] = c2.number_input(
            "Max Bars Back", 100, 5000, 
            cfg.get("max_bars_back", 1000),
            help="Размер обучающей выборки"
        )
        cfg["feature_count"] = c3.selectbox(
            "Feature Count", [2, 3, 4, 5],
            index=[2, 3, 4, 5].index(cfg.get("feature_count", 5)),
            help="Количество признаков"
        )
        
        cfg["use_dynamic_exits"] = st.checkbox(
            "Использовать динамические выходы",
            value=cfg.get("use_dynamic_exits", False),
            help="Выходы через Kernel Color Change вместо фиксированных 4 баров"
        )
        
        st.subheader("🔬 Признаки (Features)")
        feature_options = ["RSI", "WT", "CCI", "ADX"]
        current_features = cfg.get("features", ["RSI", "WT", "CCI", "ADX", "RSI"])
        current_params = cfg.get("feature_params", [[14, 1], [10, 11], [20, 1], [20, 2], [9, 1]])
        
        new_features = []
        new_params = []
        
        for i in range(5):
            c1, c2, c3 = st.columns([2, 1, 1])
            feat = c1.selectbox(
                f"Признак {i+1}", 
                feature_options,
                index=feature_options.index(current_features[i]) if i < len(current_features) and current_features[i] in feature_options else 0,
                key=f"feat_{i}"
            )
            pa = c2.number_input(
                f"Парам A", 1, 100,
                current_params[i][0] if i < len(current_params) else 14,
                key=f"pa_{i}"
            )
            pb = c3.number_input(
                f"Парам B", 1, 100,
                current_params[i][1] if i < len(current_params) else 1,
                key=f"pb_{i}"
            )
            new_features.append(feat)
            new_params.append([int(pa), int(pb)])
        
        cfg["features"] = new_features
        cfg["feature_params"] = new_params
        
        st.subheader("🛡 Фильтры")
        c1, c2 = st.columns(2)
        cfg["use_volatility_filter"] = c1.checkbox("Volatility Filter", value=cfg.get("use_volatility_filter", True))
        cfg["use_regime_filter"] = c2.checkbox("Regime Filter (Kernel)", value=cfg.get("use_regime_filter", True))
        
        c3, c4 = st.columns(2)
        cfg["use_adx_filter"] = c3.checkbox("ADX Filter", value=cfg.get("use_adx_filter", False))
        cfg["use_kernel_filter"] = c4.checkbox("Kernel Filter (для входа)", value=cfg.get("use_kernel_filter", True))
        
        c5, c6 = st.columns(2)
        cfg["use_ema_filter"] = c5.checkbox("EMA Filter (200)", value=cfg.get("use_ema_filter", False))
        cfg["use_sma_filter"] = c6.checkbox("SMA Filter (200)", value=cfg.get("use_sma_filter", False))
        
        cfg["regime_threshold"] = st.slider(
            "Regime Threshold", -1.0, 1.0,
            cfg.get("regime_threshold", -0.1), 0.1,
            help="Порог наклона kernel для regime filter"
        )
        cfg["adx_threshold"] = st.slider(
            "ADX Threshold", 10, 50,
            cfg.get("adx_threshold", 20), 1
        )
        
        st.subheader("🌊 Kernel Regression")
        kc1, kc2, kc3, kc4 = st.columns(4)
        cfg["kernel_h"] = kc1.number_input("h (Lookback)", 3, 50, cfg.get("kernel_h", 8))
        cfg["kernel_r"] = kc2.number_input("r (Relative Weight)", 0.1, 25.0, cfg.get("kernel_r", 8.0), 0.25)
        cfg["kernel_x"] = kc3.number_input("x (Regression Level)", 2, 50, cfg.get("kernel_x", 25))
        cfg["kernel_lag"] = kc4.number_input("lag", 1, 5, cfg.get("kernel_lag", 2))
        
        cfg["use_kernel_smoothing"] = st.checkbox(
            "Kernel Smoothing (crossover)",
            value=cfg.get("use_kernel_smoothing", False)
        )
        
        st.subheader("⏱ Данные")
        tf1, tf2 = st.columns(2)
        cfg["timeframe"] = tf1.selectbox(
            "Таймфрейм",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "30m", "1h", "4h", "1d"].index(cfg.get("timeframe", "15m"))
        )
        cfg["bars_depth_days"] = tf2.number_input(
            "Глубина (дней)", 7, 30, cfg.get("bars_depth_days", 30),
            help="Максимум 30 дней для MOEX ISS"
        )
        
        st.subheader("💰 Объём позиции")
        vc1, vc2 = st.columns(2)
        cfg["volume_type"] = vc1.selectbox(
            "Тип",
            ["deposit_percent", "contracts", "contract_currency"],
            index=["deposit_percent", "contracts", "contract_currency"].index(
                cfg.get("volume_type", "deposit_percent")
            )
        )
        cfg["volume_value"] = vc2.number_input(
            "Значение", 1.0, 100.0,
            cfg.get("volume_value", 5.0), 1.0
        )
        
        st.subheader("🛡 Риск-менеджмент")
        rc1, rc2, rc3 = st.columns(3)
        cfg["max_daily_orders"] = rc1.number_input(
            "Лимит заявок/день", 10, 500, cfg.get("max_daily_orders", 190)
        )
        cfg["hard_stop_loss_pct"] = rc2.slider(
            "Hard Stop Loss (%)", 0.01, 0.10,
            cfg.get("hard_stop_loss_pct", 0.02), 0.01
        )
        cfg["drawdown_reduce_pct"] = rc3.slider(
            "Reduce at DD (%)", 0.01, 0.10,
            cfg.get("drawdown_reduce_pct", 0.05), 0.01
        )
        cfg["drawdown_stop_pct"] = st.slider(
            "Stop at DD (%)", 0.05, 0.30,
            cfg.get("drawdown_stop_pct", 0.10), 0.01
        )
        
        st.subheader("🎛 Режим торговли")
        cfg["regime"] = st.selectbox(
            "Режим",
            ["On", "OnlyLong", "OnlyShort", "Off"],
            index=["On", "OnlyLong", "OnlyShort", "Off"].index(cfg.get("regime", "On")),
            help="Off = бот только наблюдает, не торгует"
        )
        
        if st.form_submit_button("💾 Сохранить настройки"):
            save_config(cfg)
            st.success("✅ Настройки сохранены! Бот применит их в следующем цикле.")


# ============================================================
# TAB: АКЦИИ
# ============================================================

elif tab == "📈 Акции":
    st.header("Список торгуемых акций (MOEX)")
    st.info("💡 Формат: `ТИКЕР@MISX` (например, SBER@MISX)")
    
    stocks = cfg.get("stocks", [])
    
    # Редактирование списка
    stocks_text = st.text_area(
        "Список акций (по одному на строку)",
        value="\n".join(stocks),
        height=400
    )
    
    if st.button("💾 Обновить список"):
        new_stocks = [s.strip() for s in stocks_text.split("\n") if s.strip()]
        # Проверка формата
        invalid = [s for s in new_stocks if "@MISX" not in s]
        if invalid:
            st.error(f"❌ Некорректный формат у: {invalid}")
        else:
            cfg["stocks"] = new_stocks
            save_config(cfg)
            st.success(f"✅ Сохранено {len(new_stocks)} акций!")
            st.rerun()
    
    st.divider()
    
    # Быстрые шаблоны
    st.subheader("🚀 Быстрые шаблоны")
    c1, c2, c3 = st.columns(3)
    
    top10 = [
        "SBER@MISX", "GAZP@MISX", "LKOH@MISX", "ROSN@MISX", "GMKN@MISX",
        "NVTK@MISX", "YNDX@MISX", "MGNT@MISX", "ALRS@MISX", "VTBR@MISX"
    ]
    blue_chips = [
        "SBER@MISX", "SBERP@MISX", "GAZP@MISX", "ROSN@MISX", "LKOH@MISX",
        "VTBR@MISX", "GMKN@MISX", "ALRS@MISX", "AFLT@MISX", "MGNT@MISX",
        "YNDX@MISX", "NVTK@MISX", "TATN@MISX", "CHMF@MISX", "NLMK@MISX"
    ]
    all_36 = [
        "SBER@MISX", "SBERP@MISX", "GAZP@MISX", "ROSN@MISX", "LKOH@MISX",
        "VTBR@MISX", "GMKN@MISX", "ALRS@MISX", "AFLT@MISX", "MGNT@MISX",
        "YNDX@MISX", "NVTK@MISX", "TATN@MISX", "CHMF@MISX", "NLMK@MISX",
        "MTSS@MISX", "MOEX@MISX", "MAGN@MISX", "PLZL@MISX", "PHOR@MISX",
        "RUAL@MISX", "RTKM@MISX", "SNGS@MISX", "SNGSP@MISX", "TRNFP@MISX",
        "TATNP@MISX", "UPRO@MISX", "MTLR@MISX", "SIBN@MISX", "BSPB@MISX",
        "FEES@MISX", "HYDR@MISX", "IRAO@MISX", "PIKK@MISX", "AFKS@MISX", "MTLRP@MISX"
    ]
    
    if c1.button("🥇 Топ-10 ликвидных"):
        cfg["stocks"] = top10
        save_config(cfg)
        st.success(f"✅ Установлено {len(top10)} акций")
        st.rerun()
    
    if c2.button("💎 Blue Chips (15)"):
        cfg["stocks"] = blue_chips
        save_config(cfg)
        st.success(f"✅ Установлено {len(blue_chips)} акций")
        st.rerun()
    
    if c3.button("📊 Все 36 акций"):
        cfg["stocks"] = all_36
        save_config(cfg)
        st.success(f"✅ Установлено {len(all_36)} акций")
        st.rerun()


# ============================================================
# TAB: СДЕЛКИ
# ============================================================

elif tab == "📋 Сделки":
    st.header("Лог сделок")
    
    if not os.path.exists(TRADES_PATH):
        st.info("📭 Пока нет сделок. Запустите бота!")
        st.stop()
    
    try:
        df = pd.read_csv(TRADES_PATH)
    except Exception as e:
        st.error(f"❌ Ошибка чтения: {e}")
        st.stop()
    
    if df.empty:
        st.info("📭 Файл trades.csv пустой")
        st.stop()
    
    st.metric("Всего сделок", len(df))
    
    # Фильтры
    c1, c2, c3 = st.columns(3)
    tickers = ["Все"] + sorted(df['ticker'].unique().tolist())
    t_filter = c1.selectbox("Тикер", tickers)
    
    sides = ["Все"] + sorted(df['side'].unique().tolist())
    side_filter = c2.selectbox("Тип операции", sides)
    
    c3.write("")
    c4 = st.columns(1)[0]
    sort_asc = c4.checkbox("Старые сверху", value=False)
    
    filtered = df.copy()
    if t_filter != "Все":
        filtered = filtered[filtered['ticker'] == t_filter]
    if side_filter != "Все":
        filtered = filtered[filtered['side'] == side_filter]
    
    if sort_asc:
        st.dataframe(filtered, width='stretch')
    else:
        st.dataframe(filtered.tail(200)[::-1], width='stretch')
    
    # Статистика
    st.subheader("📊 Статистика")
    s1, s2, s3, s4 = st.columns(4)
    
    if 'pnl' in df.columns:
        nonzero = df[df['pnl'] != 0]
        s1.metric("Общий P&L", f"{df['pnl'].sum():.2f} ₽")
        
        if len(nonzero) > 0:
            s2.metric("Winrate", f"{(nonzero['pnl'] > 0).mean()*100:.1f}%")
            s3.metric("Средний P&L", f"{nonzero['pnl'].mean():.2f} ₽")
            
            wins = nonzero[nonzero['pnl'] > 0]
            losses = nonzero[nonzero['pnl'] < 0]
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1
            s4.metric("Profit Factor", f"{avg_win / avg_loss:.2f}" if avg_loss > 0 else "∞")
    
    # Кнопка очистки
    st.divider()
    if st.button("🗑 Очистить лог сделок", type="secondary"):
        try:
            os.remove(TRADES_PATH)
            st.success("✅ trades.csv удалён")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Не удалось удалить: {e}")


# ============================================================
# TAB: ГРАФИКИ
# ============================================================

elif tab == "📉 Графики":
    st.header("Графики с Kernel Regression и ML-сигналами")
    st.caption("Rational Quadratic Kernel + метрика Минковского")
    
    stocks = cfg.get("stocks", [])
    if not stocks:
        st.warning("⚠️ Сначала настройте список акций во вкладке '📈 Акции'")
        st.stop()
    
    selected = st.selectbox("Инструмент", stocks)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        timeframe = st.selectbox(
            "Таймфрейм",
            ["15m", "1h", "4h", "1d"],
            index=0
        )
    with c2:
        days = st.number_input(
            "Дней истории", 7, 30,
            min(cfg.get("bars_depth_days", 30), 30)
        )
    
    with st.spinner(f"📥 Загрузка {selected} через MOEX ISS..."):
        try:
            client = ArenaClient(CONFIG_PATH)
            df = client.get_bars(selected, timeframe=timeframe, days=int(days))
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
    
    # Параметры
    kernel_h = cfg.get("kernel_h", 8)
    kernel_r = cfg.get("kernel_r", 8.0)
    kernel_x = cfg.get("kernel_x", 25)
    kernel_lag = cfg.get("kernel_lag", 2)
    
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    yhat1 = KernelRegression.rational_quadratic(ohlc4, kernel_h, kernel_r, kernel_x)
    yhat2 = KernelRegression.gaussian(ohlc4, max(3, kernel_h - kernel_lag), kernel_x)
    
    # ---- Цветной Kernel: разбиваем на сегменты по направлению slope ----
    def split_by_slope(series):
        """Разбивает временной ряд на сегменты по знаку прироста"""
        segments = []
        if series.isna().all():
            return segments
        start_idx = series.first_valid_index()
        if start_idx is None:
            return segments
        start_pos = list(series.index).index(start_idx)
        seg_start = start_pos
        prev_val = series.iloc[start_pos]
        for i in range(start_pos + 1, len(series)):
            if pd.isna(series.iloc[i]):
                continue
            curr_val = series.iloc[i]
            curr_is_up = curr_val > prev_val
            if i == start_pos + 1:
                prev_is_up = curr_is_up
            if curr_is_up != prev_is_up:
                segments.append((seg_start, i - 1, prev_is_up))
                seg_start = i - 1
            prev_val = curr_val
            prev_is_up = curr_is_up
        segments.append((seg_start, len(series) - 1, prev_is_up))
        return segments
    
    # ---- ML сигнал для текущего бара ----
    classifier = MinkowskiClassifier(cfg)
    eval_result = classifier.evaluate(df)
    
    # ---- Простые сигналы (prediction dots) ----
    # Сканируем kernel slope на каждом баре для визуализации направления
    kernel_signal = pd.Series(0, index=yhat1.index)
    for i in range(1, len(yhat1)):
        if pd.isna(yhat1.iloc[i]) or pd.isna(yhat1.iloc[i-1]):
            continue
        if yhat1.iloc[i] > yhat1.iloc[i-1]:
            kernel_signal.iloc[i] = 1   # bullish
        else:
            kernel_signal.iloc[i] = -1  # bearish
    
    filter_vol = Filters.filter_volatility(df, enabled=cfg.get("use_volatility_filter", True))
    filter_reg = Filters.regime_filter(df, cfg.get("regime_threshold", -0.1),
                                       kernel_h, kernel_r, cfg.get("use_regime_filter", True))
    filter_adx = Filters.filter_adx(df, 'close', 14, cfg.get("adx_threshold", 20),
                                    cfg.get("use_adx_filter", False))
    
    # ================================================================
    # Построение графика
    # ================================================================
    
    fig = go.Figure()
    
    # Свечи
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=selected,
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        showlegend=False,
    ))
    
    # Kernel RQ — цветные сегменты
    valid_mask = yhat1.notna()
    if valid_mask.any():
        valid_idx = yhat1[valid_mask].index
        segments = split_by_slope(yhat1)
        for start, end, is_up in segments:
            color = '#26a69a' if is_up else '#ef5350'
            fig.add_trace(go.Scatter(
                x=valid_idx[start:end+1],
                y=yhat1.iloc[start:end+1],
                mode='lines',
                name='Kernel RQ ↑' if is_up else 'Kernel RQ ↓',
                line=dict(color=color, width=2),
                showlegend=(start == 0 and is_up) or (start == 0 and not is_up),
                legendgroup='kernel_up' if is_up else 'kernel_down',
            ))
    
    # Gaussian Kernel (пунктир)
    if yhat2.notna().any():
        fig.add_trace(go.Scatter(
            x=df.index[yhat2.notna()],
            y=yhat2[yhat2.notna()],
            mode='lines',
            name=f'Gaussian (h={max(3, kernel_h-kernel_lag)})',
            line=dict(color='#FF9800', width=1, dash='dot')
        ))
    
    # Prediction dots — маленькие точки направления kernel на каждом баре
    bull_mask = kernel_signal == 1
    bear_mask = kernel_signal == -1
    if bull_mask.any():
        fig.add_trace(go.Scatter(
            x=df.index[bull_mask],
            y=df['low'][bull_mask] * 0.998,
            mode='markers',
            name='Kernel Bullish',
            marker=dict(symbol='circle', size=4, color='#26a69a', opacity=0.6),
            showlegend=False,
        ))
    if bear_mask.any():
        fig.add_trace(go.Scatter(
            x=df.index[bear_mask],
            y=df['high'][bear_mask] * 1.002,
            mode='markers',
            name='Kernel Bearish',
            marker=dict(symbol='circle', size=4, color='#ef5350', opacity=0.6),
            showlegend=False,
        ))
    
    # ML Signal — точки направления kernel на последних барах (2 trace вместо N)
    last_n_bars = min(80, len(df) // 2)
    sig_start = max(1, len(df) - last_n_bars)
    bull_x, bull_y = [], []
    bear_x, bear_y = [], []
    for i in range(sig_start, len(df)):
        if pd.isna(yhat1.iloc[i]) or i < 1:
            continue
        if yhat1.iloc[i] > yhat1.iloc[i-1]:
            bull_x.append(df.index[i])
            bull_y.append(df['close'].iloc[i] * 0.995)
        else:
            bear_x.append(df.index[i])
            bear_y.append(df['close'].iloc[i] * 1.005)
    if bull_x:
        fig.add_trace(go.Scatter(
            x=bull_x, y=bull_y, mode='markers',
            name='↑ Bullish', marker=dict(symbol='diamond', size=5, color='#00E676', opacity=0.5),
            showlegend=False,
        ))
    if bear_x:
        fig.add_trace(go.Scatter(
            x=bear_x, y=bear_y, mode='markers',
            name='↓ Bearish', marker=dict(symbol='diamond', size=5, color='#FF5252', opacity=0.5),
            showlegend=False,
        ))
    
    # Сигнал evaluate() для последнего бара — крупная метка
    pred = eval_result['prediction']
    signal = eval_result['signal']
    sig_color = '#00E676' if signal == 1 else '#FF5252' if signal == -1 else '#888888'
    sig_text = 'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRAL'
    sig_y = df['close'].iloc[-1] * (0.97 if signal == -1 else 1.03)
    
    fig.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[sig_y],
        mode='markers+text',
        name=f'Signal: {sig_text}',
        text=[f'{sig_text} ({pred:.1f})'],
        textposition='middle right' if signal == -1 else 'middle right',
        marker=dict(symbol='star', size=20, color=sig_color,
                    line=dict(width=2, color='white')),
    ))
    
    # Маркеры сделок из trades.csv
    if os.path.exists(TRADES_PATH):
        try:
            trades = pd.read_csv(TRADES_PATH)
            ticker_trades = trades[trades['ticker'] == selected]
            if not ticker_trades.empty:
                buys = ticker_trades[ticker_trades['side'].str.contains('BUY', na=False)]
                sells = ticker_trades[ticker_trades['side'].str.contains('SELL', na=False)]
                close_longs = ticker_trades[ticker_trades['side'].str.contains('CLOSE_LONG', na=False)]
                close_shorts = ticker_trades[ticker_trades['side'].str.contains('CLOSE_SHORT', na=False)]
                
                if not buys.empty:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(buys['timestamp']),
                        y=buys['price'],
                        mode='markers',
                        name='BUY (exec)',
                        marker=dict(symbol='triangle-up', size=14, color='#00C853',
                                   line=dict(width=1, color='white'))
                    ))
                if not sells.empty:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(sells['timestamp']),
                        y=sells['price'],
                        mode='markers',
                        name='SELL (exec)',
                        marker=dict(symbol='triangle-down', size=14, color='#FF1744',
                                   line=dict(width=1, color='white'))
                    ))
                if not close_longs.empty:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(close_longs['timestamp']),
                        y=close_longs['price'],
                        mode='markers',
                        name='CLOSE LONG',
                        marker=dict(symbol='x', size=12, color='#FF9100',
                                   line=dict(width=1, color='white'))
                    ))
                if not close_shorts.empty:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(close_shorts['timestamp']),
                        y=close_shorts['price'],
                        mode='markers',
                        name='CLOSE SHORT',
                        marker=dict(symbol='x', size=12, color='#00BCD4',
                                   line=dict(width=1, color='white'))
                    ))
        except Exception:
            pass
    
    # Layout
    fig.update_layout(
        title=f"{selected} — {timeframe} ({len(df)} свечей) | Pred={pred:.1f} Signal={sig_text}",
        yaxis_title="Цена",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=650,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # ================================================================
    # ML PANEL: детальная информация о текущем сигнале
    # ================================================================
    
    st.subheader("🧠 ML-метрики текущего бара")
    
    col_pred, col_sig, col_filt, col_nn = st.columns(4)
    
    # Prediction
    pred_color = "🟢" if pred > 0 else "🔴" if pred < 0 else "⚪"
    col_pred.metric(f"{pred_color} Prediction (сумма лейблов)", f"{pred:.1f}",
                    help=f"Сумма меток {classifier.neighbors_count} соседей: >0 = LONG, <0 = SHORT")
    
    # Signal
    sig_label = {1: "LONG ↑", -1: "SHORT ↓", 0: "NEUTRAL —"}.get(signal, "—")
    col_sig.metric("📡 Сигнал", sig_label,
                   help=f"signal = prediction ({pred:.1f}) после проверки фильтров")
    
    # Filters
    filters_ok = sum([filter_vol, filter_reg, filter_adx])
    filters_total = 3
    col_filt.metric("🛡 Фильтры", f"{filters_ok}/{filters_total} ✅" if filters_ok == filters_total else f"{filters_ok}/{filters_total} ❌",
                    help=f"Volatility={'✅' if filter_vol else '❌'} Regime={'✅' if filter_reg else '❌'} ADX={'✅' if filter_adx else '❌'}")
    
    # Kernel state
    is_bull_rate = eval_result.get('is_bullish_rate', False)
    is_bear_rate = eval_result.get('is_bearish_rate', False)
    kernel_trend = "🟢 UP" if is_bull_rate else "🔴 DOWN" if is_bear_rate else "⚪ FLAT"
    col_nn.metric("🌊 Kernel Trend", kernel_trend,
                  help=f"is_bullish_rate={is_bull_rate}, is_bearish_rate={is_bear_rate}")
    
    # ---- Feature Values ----
    with st.expander("📊 Значения признаков (Features)", expanded=True):
        feat_cols = st.columns(classifier.feature_count)
        feature_series = []
        for i in range(classifier.feature_count):
            feat_name = classifier.features[i]
            param_a, param_b = classifier.feature_params[i]
            series = classifier._get_feature_series(df, feat_name, param_a, param_b)
            feature_series.append(series)
            val = series.iloc[-1] if not pd.isna(series.iloc[-1]) else 0.5
            # Цвет: green=high (>0.6), red=low (<0.4), gray=mid
            val_color = "#00E676" if val > 0.6 else "#FF5252" if val < 0.4 else "#888888"
            feat_cols[i].metric(
                f"{feat_name}({param_a},{param_b})",
                f"{val:.4f}",
                help=f"Нормализованное значение [0..1]"
            )
    
    # ---- Filter Status ----
    with st.expander("🛡 Статус фильтров", expanded=True):
        fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
        
        # EMA filter
        ema_up = eval_result.get('is_ema_uptrend', True)
        ema_down = eval_result.get('is_ema_downtrend', True)
        ema_enabled = cfg.get("use_ema_filter", False)
        if ema_enabled:
            ema_status = "✅" if eval_result.get('is_ema_uptrend', False) else "❌"
            ema_text = f"{ema_status}\nclose > EMA(200)"
        else:
            ema_status = "⏭"
            ema_text = "OFF"
        fc1.metric("EMA(200)", ema_text, help=f"EMA фильтр {'включён' if ema_enabled else 'выключен'}")
        
        # SMA filter
        sma_enabled = cfg.get("use_sma_filter", False)
        sma_status = "✅" if eval_result.get('is_sma_uptrend', True) else "❌"
        fc2.metric("SMA(200)", f"{'⏭' if not sma_enabled else sma_status}\nclose > SMA(200)" if not sma_enabled else f"{sma_status}\nclose > SMA(200)",
                   help=f"SMA фильтр {'включён' if sma_enabled else 'выключен'}")
        
        # Volatility filter
        vol_pass = "✅" if filter_vol else "❌"
        fc3.metric("Volatility", f"{vol_pass}\nTR > SMA(TR,20)*10" if filter_vol else f"{vol_pass}\nTR ≤ SMA(TR,20)*10",
                   help=f"volatility_filter={'включён' if cfg.get('use_volatility_filter', True) else 'выключен'}")
        
        # Regime filter
        reg_pass = "✅" if filter_reg else "❌"
        fc4.metric("Regime", f"{reg_pass}\nslope > {cfg.get('regime_threshold', -0.1)}" if filter_reg else f"{reg_pass}\nslope ≤ {cfg.get('regime_threshold', -0.1)}",
                   help=f"regime_filter={'включён' if cfg.get('use_regime_filter', True) else 'выключен'}")
        
        # ADX filter
        adx_pass = "✅" if filter_adx else "❌"
        fc5.metric("ADX", f"{adx_pass}\nADX > {cfg.get('adx_threshold', 20)}" if filter_adx else f"{adx_pass}\nADX ≤ {cfg.get('adx_threshold', 20)}",
                   help=f"adx_filter={'включён' if cfg.get('use_adx_filter', False) else 'выключен'}")
        
        # Kernel filter (для входа)
        kernel_enabled = cfg.get("use_kernel_filter", True)
        if kernel_enabled:
            kernel_filt_pass = "✅" if (is_bull_rate or is_bear_rate) else "❌"
        else:
            kernel_filt_pass = "⏭"
        fc6.metric("Kernel (вход)", f"{kernel_filt_pass}\n{'rate' if not cfg.get('use_kernel_smoothing', False) else 'smooth'}" if kernel_enabled else f"{kernel_filt_pass}\nOFF",
                   help=f"kernel_filter={'включён' if kernel_enabled else 'выключен'}")
    
    # ---- Neighbors Info ----
    with st.expander(f"🔢 Соседи (k={classifier.neighbors_count})", expanded=False):
        st.info("ANN выбрал ближайших соседей по метрике Минковского. Prediction = сумма их меток (+1 LONG, -1 SHORT).")
        st.write(f"**Prediction:** {pred:.1f} (из {classifier.neighbors_count} соседей)")
        bull_count = int((pred + classifier.neighbors_count) / 2) if pred >= -classifier.neighbors_count and pred <= classifier.neighbors_count else 0
        bear_count = classifier.neighbors_count - bull_count
        st.write(f"🟢 Бычьих: {bull_count} | 🔴 Медвежьих: {bear_count}")
        if classifier.neighbors_count > 0:
            st.progress(bull_count / classifier.neighbors_count, text="Bullish ratio")
    
    # ---- Entry/Exit Rules Check ----
    with st.expander("🚦 Проверка правил входа/выхода", expanded=False):
        # Упрощённая проверка для текущего бара (без истории сигналов)
        is_long_signal = signal == 1
        is_short_signal = signal == -1
        is_bullish_rate = eval_result.get('is_bullish_rate', False)
        is_bearish_rate = eval_result.get('is_bearish_rate', False)
        
        ema_ok = ema_up if ema_enabled else True
        sma_ok = eval_result.get('is_sma_uptrend', True) if cfg.get("use_sma_filter", False) else True
        kernel_ok = is_bullish_rate if kernel_enabled and not cfg.get("use_kernel_smoothing", False) else True
        
        st.write("**start_long** = is_new_buy AND is_bullish AND ema_up AND sma_up")
        st.write(f"├ signal==1: {'✅' if is_long_signal else '❌'} | is_bullish_rate: {'✅' if is_bullish_rate else '❌'} | ema_ok: {'✅' if ema_ok else '❌'} | sma_ok: {'✅' if sma_ok else '❌'}")
        st.write(f"→ {'🟢 МОЖЕМ ВОЙТИ В LONG' if (is_long_signal and is_bullish and kernel_ok and ema_ok and sma_ok) else '⛔ НЕТ СИГНАЛА НА LONG'}")
        
        st.write("")
        st.write("**start_short** = is_new_sell AND is_bearish AND ema_down AND sma_down")
        ema_down_ok = eval_result.get('is_ema_downtrend', True) if ema_enabled else True
        sma_down_ok = eval_result.get('is_sma_downtrend', True) if cfg.get("use_sma_filter", False) else True
        kernel_bear_ok = is_bearish_rate if kernel_enabled and not cfg.get("use_kernel_smoothing", False) else True
        st.write(f"├ signal==-1: {'✅' if is_short_signal else '❌'} | is_bearish_rate: {'✅' if is_bearish_rate else '❌'} | ema_down: {'✅' if ema_down_ok else '❌'} | sma_down: {'✅' if sma_down_ok else '❌'}")
        st.write(f"→ {'🔴 МОЖЕМ ВОЙТИ В SHORT' if (is_short_signal and is_bearish and kernel_bear_ok and ema_down_ok and sma_down_ok) else '⛔ НЕТ СИГНАЛА НА SHORT'}")
    
    # ---- Статистика по инструменту ----
    st.divider()
    st.subheader("📊 Статистика")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Последняя цена", f"{df['close'].iloc[-1]:.2f} ₽")
    
    last_kernel = yhat1.iloc[-1]
    if not pd.isna(last_kernel):
        sc2.metric("Kernel RQ", f"{last_kernel:.2f} ₽")
        prev_kernel = yhat1.iloc[-2] if len(yhat1) > 1 and not pd.isna(yhat1.iloc[-2]) else last_kernel
        trend = "🟢 UP" if last_kernel > prev_kernel else "🔴 DOWN"
        sc3.metric("Тренд Kernel", trend)
    else:
        sc2.metric("Kernel RQ", "—")
        sc3.metric("Тренд Kernel", "—")
    
    sc4.metric("Свечей", len(df))
    
    if os.path.exists(TRADES_PATH):
        try:
            trades_df = pd.read_csv(TRADES_PATH)
            ticker_trades = trades_df[trades_df['ticker'] == selected]
            sc5.metric("Сделок по инструменту", len(ticker_trades))
        except Exception:
            sc5.metric("Сделок по инструменту", 0)
    else:
        sc5.metric("Сделок по инструменту", 0)


# ============================================================
# TAB: БЕЗОПАСНОСТЬ
# ============================================================

elif tab == "🚨 Безопасность":
    st.header("🚨 Аварийное управление")
    st.warning("⚠️ Используйте эти кнопки только в экстренных случаях!")
    
    client = ArenaClient(CONFIG_PATH)
    
    st.subheader("📋 Текущие открытые позиции")
    with st.spinner("Загрузка позиций..."):
        positions = client.get_positions()
    
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
                                res = client.place_market_order(sym, side, qty)
                                
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
                            if os.path.exists("positions_state.json"):
                                os.remove("positions_state.json")
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
        with open("stop.flag", "w") as f:
            f.write("1")
        st.success("✅ stop.flag создан. Бот остановится в течение минуты.")
    
    # Информация
    with st.expander("ℹ️ Как это работает?"):
        st.markdown("""
        **Закрытие по рынку:**
        Отправляет встречные рыночные ордера на все открытые позиции.
        Исполнение происходит мгновенно по текущей рыночной цене.
        
        **stop.flag:**
        Специальный файл-сигнал. Бот проверяет его наличие каждую секунду.
        При обнаружении — завершает работу корректно.
        
        **Все действия логируются** в `trades.csv` для последующего анализа.
        """)