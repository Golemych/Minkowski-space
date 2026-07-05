import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from arena import SIDE_BUY, SIDE_SELL, ArenaClient
from indicators import Indicators, ContangoFilter, ContangoCalculator, normalize_df

BASE_DIR = Path(__file__).parent
PAIRS_PATH = BASE_DIR / "pairs.json"
CACHE_DIR = BASE_DIR / "cache"

STRATEGY = "bollinger"
REGIME = "On"
TIMEFRAME = "15m"
BARS_DEPTH_DAYS = 7

BOLLINGER_LENGTH = 230
BOLLINGER_DEVIATION = 2.1

KELTNER_EMA_LENGTH = 150
KELTNER_ATR_LENGTH = 24
KELTNER_DEVIATION = 3.9

CONTANGO_FILTER_REGIME = "On_MOEXStocksAuto"
CONTANGO_FILTER_COUNT = 5
CONTANGO_STAGE_LONG = 1
CONTANGO_STAGE_SHORT = 2

VOLUME_TYPE = "deposit_percent"
VOLUME_VALUE = 15.0
TRADE_ASSET = "Prime"

ICEBERG_COUNT = 1

MAX_DAILY_ORDERS = 190
HARD_STOP_LOSS_PCT = 0.02
DRAWDOWN_REDUCE_PCT = 0.05
DRAWDOWN_STOP_PCT = 0.10

TRADE_START = dtime(10, 5)
TRADE_END = dtime(18, 30)

MIN_EXPIRATION_DAYS = 3
MAX_EXPIRATION_DAYS = 100

LOOP_SLEEP_SEC = 60

DECIMALS_MAP = {
    "SBER": 2, "SBERP": 2, "GAZP": 2, "ROSN": 2,
    "LKOH": 2, "VTBR": 2, "GMKN": 2, "ALRS": 2,
    "AFLT": 2, "MGNT": 2,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("futures_bot")

MSK = timezone(timedelta(hours=3))


def _is_trading_time() -> bool:
    now = datetime.now(MSK)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return TRADE_START <= t < TRADE_END


def _get_expiration(futures_symbol: str) -> Optional[datetime]:
    import re
    clean = futures_symbol.split('@')[0].upper()
    m = re.search(r'-(\d{1,2})\.(\d{2})$', clean)
    if m:
        month, year = int(m.group(1)), 2000 + int(m.group(2))
        return datetime(year, month, 15, 23, 59, 59)
    if len(clean) >= 4:
        month_map = {"H": 3, "M": 6, "U": 9, "Z": 12}
        letter, dig = clean[-2], clean[-1]
        if letter in month_map:
            return datetime(2020 + int(dig), month_map[letter], 15, 23, 59, 59)
    return None


def _days_to_expiration(futures_symbol: str) -> Optional[float]:
    exp = _get_expiration(futures_symbol)
    if exp is None:
        return None
    delta = exp - datetime.now()
    return delta.total_seconds() / (24 * 3600)


def _check_expiration_range(futures_symbol: str) -> bool:
    days = _days_to_expiration(futures_symbol)
    if days is None:
        return True
    if days < MIN_EXPIRATION_DAYS or days > MAX_EXPIRATION_DAYS:
        log.info(f"Expiration {futures_symbol}: {days:.1f}d out of range")
        return False
    return True


def _check_expiration_exit(futures_symbol: str) -> bool:
    days = _days_to_expiration(futures_symbol)
    if days is None:
        return False
    return days < MIN_EXPIRATION_DAYS


def load_pairs() -> list[dict]:
    if not PAIRS_PATH.exists():
        log.error(f"pairs.json not found at {PAIRS_PATH}")
        return []
    with open(PAIRS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [{"ticker": k, **v} for k, v in raw.items()]
    return []


def _enabled_pairs(pairs: list[dict]) -> list[dict]:
    return [p for p in pairs if p.get("enabled", True)]


def calc_contango_coeff(ticker: str, decimals: int,
                        now: Optional[datetime] = None) -> float:
    if now is None:
        now = datetime.now()
    ticker_upper = ticker.upper()

    if "VTB" in ticker_upper or "VTBR" in ticker_upper:
        if now.year < 2024:
            return 20
        if now.year == 2024 and now.month < 7:
            return 20
        if now.year == 2024 and now.month == 7 and now.day < 15:
            return 20
        return 100

    if "GMKN" in ticker_upper:
        if now.year < 2024:
            return 100
        if now.year == 2024 and now.month < 4:
            return 100
        if now.year == 2024 and now.month == 4 and now.day < 4:
            return 100
        return 10

    coeff = 1.0
    for _ in range(decimals):
        coeff *= 10
    return coeff


def calc_signal(
    ticker: str,
    strategy: str,
    futures_symbol: str,
    spot_price: float,
    futures_price: float,
    futures_df: pd.DataFrame,
    contango_filter: ContangoFilter,
    contango_coeff: float,
) -> Optional[dict]:
    if not _is_trading_time():
        return None
    if not _check_expiration_range(futures_symbol):
        return None

    futures_df = normalize_df(futures_df)

    if strategy == "keltner":
        min_len = max(KELTNER_EMA_LENGTH, KELTNER_ATR_LENGTH)
        if len(futures_df) < min_len:
            return None
        k_u, k_m, k_l = Indicators.keltner_channel(
            futures_df, KELTNER_EMA_LENGTH, KELTNER_ATR_LENGTH, KELTNER_DEVIATION
        )
        upper = float(k_u.iloc[-1])
        lower = float(k_l.iloc[-1])
        if any(pd.isna(x) for x in (upper, lower)):
            return None
        is_long = futures_price > upper
        is_short = futures_price < lower
        reason = f"KC 上={upper:.2f} 下={lower:.2f}"
    else:
        min_len = BOLLINGER_LENGTH
        if len(futures_df) < min_len:
            return None
        bb_u, bb_m, bb_l = Indicators.bollinger_bands(
            futures_df, BOLLINGER_LENGTH, BOLLINGER_DEVIATION
        )
        upper = float(bb_u.iloc[-1])
        lower = float(bb_l.iloc[-1])
        if any(pd.isna(x) for x in (upper, lower)):
            return None
        is_long = futures_price > upper
        is_short = futures_price < lower
        reason = f"BB 上={upper:.2f} 下={lower:.2f}"

    if not is_long and not is_short:
        return None

    contango_pct = ContangoCalculator.calculate(spot_price, futures_price, contango_coeff)
    stage = contango_filter.get_stage(ticker, CONTANGO_FILTER_COUNT)

    if is_long:
        if REGIME == "Off" or REGIME == "OnlyShort":
            return None
        if stage != CONTANGO_STAGE_LONG:
            return None
        return {
            "action": "BUY", "reason": reason,
            "price": futures_price,
            "contango_pct": contango_pct, "contango_stage": stage,
        }

    if is_short:
        if REGIME == "Off" or REGIME == "OnlyLong":
            return None
        if stage != CONTANGO_STAGE_SHORT:
            return None
        return {
            "action": "SELL", "reason": reason,
            "price": futures_price,
            "contango_pct": contango_pct, "contango_stage": stage,
        }

    return None


def check_exit(
    ticker: str,
    strategy: str,
    position_side: str,
    futures_df: pd.DataFrame,
    futures_price: float,
    futures_symbol: str,
) -> bool:
    if _check_expiration_exit(futures_symbol):
        log.info(f"{ticker}: exit by expiration")
        return True

    futures_df = normalize_df(futures_df)

    if strategy == "keltner":
        min_len = max(KELTNER_EMA_LENGTH, KELTNER_ATR_LENGTH)
        if len(futures_df) < min_len:
            return False
        k_u, _, k_l = Indicators.keltner_channel(
            futures_df, KELTNER_EMA_LENGTH, KELTNER_ATR_LENGTH, KELTNER_DEVIATION
        )
        upper = float(k_u.iloc[-1])
        lower = float(k_l.iloc[-1])
        if any(pd.isna(x) for x in (upper, lower)):
            return False
        if position_side == "BUY" and futures_price < lower:
            log.info(f"{ticker}: exit LONG (price={futures_price:.2f} < KC下={lower:.2f})")
            return True
        if position_side == "SELL" and futures_price > upper:
            log.info(f"{ticker}: exit SHORT (price={futures_price:.2f} > KC上={upper:.2f})")
            return True
    else:
        min_len = BOLLINGER_LENGTH
        if len(futures_df) < min_len:
            return False
        bb_u, _, bb_l = Indicators.bollinger_bands(
            futures_df, BOLLINGER_LENGTH, BOLLINGER_DEVIATION
        )
        upper = float(bb_u.iloc[-1])
        lower = float(bb_l.iloc[-1])
        if any(pd.isna(x) for x in (upper, lower)):
            return False
        if position_side == "BUY" and futures_price < lower:
            log.info(f"{ticker}: exit LONG (price={futures_price:.2f} < BB下={lower:.2f})")
            return True
        if position_side == "SELL" and futures_price > upper:
            log.info(f"{ticker}: exit SHORT (price={futures_price:.2f} > BB上={upper:.2f})")
            return True

    return False


@dataclass
class Position:
    symbol: str
    side: str
    qty: int
    entry_price: float
    entry_time: datetime
    entry_comm: float


@dataclass
class State:
    positions: dict = field(default_factory=dict)
    total_pnl: float = 0.0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    daily_orders: int = 0
    last_reset_date: Optional[datetime] = None
    is_stopped: bool = False
    initial_equity: Optional[float] = None
    contango_filter: ContangoFilter = field(default_factory=ContangoFilter)
    pairs_cache: list[dict] = field(default_factory=list)


def _calc_volume(price: float, balance: float, lot: int = 1) -> int:
    if balance == 0 or price == 0:
        return 1
    if VOLUME_TYPE == "deposit_percent":
        money_on_pos = balance * (VOLUME_VALUE / 100)
        qty = money_on_pos / price / lot
        return max(1, int(round(qty)))
    elif VOLUME_TYPE == "contracts":
        return int(VOLUME_VALUE)
    elif VOLUME_TYPE == "contract_currency":
        return max(1, int(round(VOLUME_VALUE / price)))
    return 1


async def open_position(
    arena: ArenaClient,
    state: State,
    signal: dict,
    futures_symbol: str,
    balance: float,
    drawdown_reduce: bool,
) -> bool:
    price = signal["price"]
    qty = _calc_volume(price, balance)
    if drawdown_reduce:
        qty = max(1, qty // 2)

    side = SIDE_BUY if signal["action"] == "BUY" else SIDE_SELL
    log.info(f"OPEN  {signal['action']} {futures_symbol} ×{qty} @ {price:.4f}  [{signal['reason']}]  "
             f"contango={signal['contango_pct']:.3f}%  iceberg={ICEBERG_COUNT}")
    result = await arena.place_order(futures_symbol, side, qty, iceberg_count=ICEBERG_COUNT)

    if result:
        state.positions[futures_symbol] = Position(
            symbol=futures_symbol,
            side=signal["action"],
            qty=qty,
            entry_price=result.execution_price,
            entry_time=datetime.now(MSK),
            entry_comm=result.commission,
        )
        state.daily_orders += 1
        log.info(f"Opened {result.execution_price:.4f}  comm={result.commission:.4f}")
        return True
    else:
        log.error(f"Failed to open {futures_symbol}")
        return False


async def close_position(
    arena: ArenaClient,
    state: State,
    futures_symbol: str,
    reason: str,
    pos: Position,
    current_price: float,
) -> bool:
    side = SIDE_SELL if pos.side == "BUY" else SIDE_BUY
    log.info(f"CLOSE [{reason}]  {pos.side} {futures_symbol} ×{pos.qty}")
    result = await arena.place_order(futures_symbol, side, pos.qty, iceberg_count=ICEBERG_COUNT)

    if result:
        gross_pnl = (result.execution_price - pos.entry_price) * pos.qty
        if pos.side == "SELL":
            gross_pnl = -gross_pnl
        total_comm = pos.entry_comm + result.commission
        net_pnl = gross_pnl - total_comm

        state.total_pnl += net_pnl
        state.trade_count += 1
        if net_pnl > 0:
            state.wins += 1
        else:
            state.losses += 1
        state.daily_orders += 1
        del state.positions[futures_symbol]

        held = datetime.now(MSK) - pos.entry_time
        log.info(
            f"Closed  pnl={net_pnl:+.2f}  held={str(held).split('.')[0]}  "
            f"total_pnl={state.total_pnl:+.2f}  trades={state.trade_count} "
            f"(W{state.wins}/L{state.losses})"
        )
        return True
    else:
        log.error(f"Failed to close {futures_symbol}")
        return False


def _check_drawdown(state: State, current_equity: float) -> str:
    if state.initial_equity is None or state.initial_equity == 0 or current_equity == 0:
        return "OK"
    dd = max(0, (state.initial_equity - current_equity) / state.initial_equity)
    if dd >= DRAWDOWN_STOP_PCT:
        state.is_stopped = True
        log.error(f"DRAWDOWN {dd*100:.2f}% >= {DRAWDOWN_STOP_PCT*100}% — STOPPED")
        return "STOP"
    if dd >= DRAWDOWN_REDUCE_PCT:
        log.warning(f"Drawdown {dd*100:.2f}% >= {DRAWDOWN_REDUCE_PCT*100}% — reduce volume")
        return "REDUCE"
    return "OK"


def _print_summary(state: State):
    log.info(
        "═══ FINAL SUMMARY ═══\n"
        f"trades={state.trade_count}  wins={state.wins}  losses={state.losses}\n"
        f"total_pnl={state.total_pnl:+.2f}  open_positions={len(state.positions)}"
    )


async def run_strategy(
    arena: ArenaClient,
    state: State,
    stop_event: asyncio.Event,
) -> None:
    pairs = _enabled_pairs(load_pairs())
    if not pairs:
        log.error("No enabled pairs in pairs.json")
        return

    log.info(f"Active pairs: {[p['ticker'] for p in pairs]}")

    while not stop_event.is_set():
        try:
            pairs = _enabled_pairs(load_pairs())

            today = datetime.now(MSK).date()
            if state.last_reset_date != today:
                state.daily_orders = 0
                state.last_reset_date = today
                log.info(f"Daily reset: orders=0")

            balance = await arena.get_balance()
            equity = await arena.get_equity()
            if state.initial_equity is None:
                state.initial_equity = equity
                log.info(f"Initial equity: {equity:,.2f}")

            dd_status = _check_drawdown(state, equity)
            if dd_status == "STOP":
                await asyncio.sleep(LOOP_SLEEP_SEC)
                continue

            arena_positions = await arena.get_positions()
            arena_pos_by_sym = {p["symbol"]: p for p in arena_positions}

            for sym in list(state.positions.keys()):
                if sym not in arena_pos_by_sym:
                    log.warning(f"Position {sym} no longer in Arena, removing from state")
                    del state.positions[sym]

            state.contango_filter = ContangoFilter()

            contango_data = []
            for pair in pairs:
                ticker = pair["ticker"]
                spot_symbol = pair.get("spot")
                futures_symbol = pair.get("futures")
                if not spot_symbol or not futures_symbol:
                    continue
                spot_df = await arena.get_bars(spot_symbol, TIMEFRAME, BARS_DEPTH_DAYS)
                futures_df = await arena.get_bars(futures_symbol, TIMEFRAME, BARS_DEPTH_DAYS)
                if spot_df is None or futures_df is None:
                    log.warning(f"{ticker}: no data"); continue
                spot_price = float(spot_df["close"].iloc[-1])
                futures_price = float(futures_df["close"].iloc[-1])

                coeff = pair.get("contango_coeff")
                if coeff is None:
                    if CONTANGO_FILTER_REGIME == "On_MOEXStocksAuto":
                        decimals = DECIMALS_MAP.get(ticker.upper(), 2)
                        coeff = calc_contango_coeff(ticker, decimals)
                    elif CONTANGO_FILTER_REGIME == "On_Manual":
                        coeff = float(pair.get("manual_coeff", 100))
                    else:
                        coeff = 100.0
                else:
                    coeff = float(coeff)

                spot_ask, _ = await arena.get_best_bid_ask(spot_symbol)
                futures_bid, _ = await arena.get_best_bid_ask(futures_symbol)
                if spot_ask is not None and futures_bid is not None:
                    cp = ContangoCalculator.calculate(spot_ask, futures_bid, coeff)
                else:
                    cp = ContangoCalculator.calculate(spot_price, futures_price, coeff)
                state.contango_filter.update(ticker, cp)
                contango_data.append((pair, ticker, spot_symbol, futures_symbol,
                                       spot_df, futures_df, spot_price, futures_price, coeff))

            for (pair, ticker, spot_symbol, futures_symbol, spot_df, futures_df,
                 spot_price, futures_price, coeff) in contango_data:
                strategy = pair.get("strategy", STRATEGY)

                if futures_symbol not in state.positions:
                    signal = calc_signal(
                        ticker, strategy, futures_symbol, spot_price, futures_price,
                        futures_df, state.contango_filter, coeff,
                    )
                    if signal:
                        await open_position(arena, state, signal, futures_symbol,
                                            balance, dd_status == "REDUCE")
                else:
                    pos = state.positions[futures_symbol]
                    if check_exit(ticker, strategy, pos.side, futures_df,
                                  futures_price, futures_symbol):
                        await close_position(arena, state, futures_symbol,
                                             "Exit by signal", pos, futures_price)

            if state.initial_equity and state.initial_equity > 0:
                for sym, apos in arena_pos_by_sym.items():
                    pnl = apos.get("unrealized_pnl", 0)
                    pnl_pct = pnl / state.initial_equity
                    if pnl_pct <= -HARD_STOP_LOSS_PCT:
                        pos = state.positions.get(sym)
                        if pos:
                            log.error(f"HARD STOP {sym}: P&L {pnl:.2f} ({pnl_pct*100:.2f}%)")
                            await close_position(arena, state, sym,
                                                 f"Hard Stop {pnl_pct*100:.2f}%", pos, 0.0)

            log.info(
                f"Cycle done: equity={equity:,.0f}  positions={len(state.positions)}  "
                f"orders_today={state.daily_orders}  total_pnl={state.total_pnl:+.2f}"
            )

        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error(f"Cycle error: {exc}", exc_info=True)

        for _ in range(LOOP_SLEEP_SEC):
            if stop_event.is_set():
                return
            await asyncio.sleep(1)


async def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    api_token = os.environ.get("ARENA_API_TOKEN")
    account_id_str = os.environ.get("ARENA_ACCOUNT_ID")

    if not api_token or not account_id_str:
        raise RuntimeError("ARENA_API_TOKEN and ARENA_ACCOUNT_ID required in .env")

    account_id = int(account_id_str)

    log.info(
        f"Futures Strategy  strategy={STRATEGY}\n"
        f"Bollinger: {BOLLINGER_LENGTH}/{BOLLINGER_DEVIATION}  "
        f"Keltner: {KELTNER_EMA_LENGTH}/{KELTNER_ATR_LENGTH}/{KELTNER_DEVIATION}\n"
        f"Contango: regime={CONTANGO_FILTER_REGIME}  count={CONTANGO_FILTER_COUNT}\n"
        f"Regime={REGIME}  Vol={VOLUME_TYPE}={VOLUME_VALUE}  "
        f"Iceberg={ICEBERG_COUNT}  MaxOrders={MAX_DAILY_ORDERS}/d\n"
        f"Trade: {TRADE_START.strftime('%H:%M')}–{TRADE_END.strftime('%H:%M')} MSK  "
        f"Expiration: {MIN_EXPIRATION_DAYS}d–{MAX_EXPIRATION_DAYS}d"
    )

    state = State()
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)
    except NotImplementedError:
        pass

    try:
        async with ArenaClient(api_token, account_id, cache_dir=CACHE_DIR) as arena:
            await run_strategy(arena, state, stop_event)
    finally:
        _print_summary(state)


if __name__ == "__main__":
    asyncio.run(main())
