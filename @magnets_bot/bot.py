import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
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
    last_signals: List[str] = None
    last_errors: List[str] = None
    stats: Dict[str, int] = None
    positions_owners: Dict[str, str] = None
    last_update: str = None

async def _load_account_state(state: State):
    client = ArenaClient(os.environ.get("ARENA_API_TOKEN"), int(os.environ.get("ARENA_ACCOUNT_ID")), BASE_DIR / "cache")
    info = await client.get_account()
    positions = await client.get_positions()

    if info:
        state.status = "IDLE"
        state.balance = float(info.get("available_cash", {}).get("value", 0))
        state.equity = float(info.get("equity", {}).get("value", 0))
        state.cash = float(info.get("cash", {}).get("value", 0))
        state.available_cash = float(info.get("available_cash", {}).get("value", 0))
        state.open_positions = len(positions)
        state.last_update = datetime.now().isoformat()

async def run_bot():
    # Load configuration
    config = load_config()
    if not config:
        print("❌ Configuration error. Check config.json.")
        return

    # Initialize Arena client
    client = ArenaClient(config.get("api_secret"), config.get("account_id"), BASE_DIR / "cache")

    # Initialize strategy manager
    classifier = MinkowskiClassifier()
    strategy_manager = StrategyManager(classifier)

    # Load state
    state = load_state()
    if not state:
        await _load_account_state(state)
        save_state(state)

    while True:
        try:
            # Check trading hours
            now = datetime.now()
            trading_hours = (now.hour >= 10 and now.hour < 18) or (now.hour == 18 and now.minute < 30)
            if not trading_hours:
                state.status = "IDLE"
                save_state(state)
                await asyncio.sleep(60)
                continue

            # Fetch data
            symbols = [s["symbol"] for s in config.get("stocks", [])]
            data = {}
            for symbol in symbols:
                bars = await client.get_bars(symbol, timeframe="15m", days=30)
                if bars is not None:
                    data[symbol] = normalize_df(bars)

            # Generate signals
            signals = strategy_manager.generate_signals(data)
            state.last_signals = [f"{sym}: {sig}" for sym, sig in signals.items()]

            # Execute trades
            for symbol, signal in signals.items():
                if signal == "BUY":
                    res = await client.place_order(symbol, side="SIDE_BUY", qty=1)
                    if res:
                        state.stats["BUY"] += 1
                        state.positions_owners[symbol] = STRATEGY_NAME
                elif signal == "SELL":
                    res = await client.place_order(symbol, side="SIDE_SELL", qty=int(state.positions_owners.get(symbol, 0)))
                    if res:
                        state.stats["SELL"] += 1
                        del state.positions_owners[symbol]

            # Update state
            state.daily_orders += len(signals)
            state.last_update = datetime.now().isoformat()
            save_state(state)

            # Check drawdown
            if state.equity < state.balance * (1 - config.get("drawdown_threshold", 0.05)):
                state.status = "LIMIT_REACHED"
                save_state(state)
                print(f"⚠️ Drawdown limit reached: {state.drawdown_pct:.2f}%")
                break

            # Sleep for a while before the next iteration
            await asyncio.sleep(60 * 15)  # 15 minutes
        except Exception as e:
            state.last_errors.append(f"{datetime.now().isoformat()}: {str(e)}")
            save_state(state)
            print(f"❌ Error: {e}")
            await asyncio.sleep(60 * 5)  # 5 minutes

    state.status = "STOPPED"
    save_state(state)
    print("🤖 Bot stopped.")

if __name__ == "__main__":
    asyncio.run(run_bot())