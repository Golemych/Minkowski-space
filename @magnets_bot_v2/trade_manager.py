"""
Исполнение сделок: открытие/закрытие позиций, риск-менеджмент.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from arena_client import ArenaClient

logger = logging.getLogger(__name__)


class TradeExecutor:
    def __init__(self, config: dict, arena: ArenaClient):
        self.config = config
        self.arena = arena
        self.initial_balance = None
        self.daily_orders = 0
        self.last_reset_date = None
        self.position_owners: Dict = {}

    def load_positions(self, path: str = "positions_state.json"):
        import json, os
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.position_owners = json.load(f)
            except Exception:
                self.position_owners = {}

    def save_positions(self, path: str = "positions_state.json"):
        import json
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.position_owners, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения positions_state.json: {e}")

    def calculate_volume(self, price: float) -> int:
        balance = self.arena.get_balance()
        if balance == 0 or price == 0:
            return 1
        volume_type = self.config.get("volume_type", "deposit_percent")
        volume_value = self.config.get("volume_value", 5.0)
        if volume_type == "deposit_percent":
            amount = balance * (volume_value / 100)
            return max(1, int(amount / price))
        elif volume_type == "contracts":
            return int(volume_value)
        elif volume_type == "contract_currency":
            return max(1, int(volume_value / price))
        return 1

    def check_daily_limit(self) -> bool:
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_orders = 0
            self.last_reset_date = today
        max_orders = self.config.get("max_daily_orders", 190)
        if self.daily_orders >= max_orders:
            logger.warning(f"Дневной лимит: {self.daily_orders}/{max_orders}")
            return False
        return True

    def check_drawdown(self) -> str:
        if self.initial_balance is None or self.initial_balance == 0:
            return "OK"
        equity = self.arena.get_equity()
        if equity == 0:
            return "OK"
        drawdown_pct = max(0, (self.initial_balance - equity) / self.initial_balance)
        stop_pct = self.config.get("drawdown_stop_pct", 0.10)
        reduce_pct = self.config.get("drawdown_reduce_pct", 0.05)
        if drawdown_pct >= stop_pct:
            logger.error(f"ПРОСАДКА {drawdown_pct*100:.2f}% >= {stop_pct*100}%. ОСТАНОВКА!")
            return "STOP"
        elif drawdown_pct >= reduce_pct:
            logger.warning(f"Просадка {drawdown_pct*100:.2f}% >= {reduce_pct*100}%. Уменьшаем объём")
            return "REDUCE"
        return "OK"

    def check_hard_stop(self, positions: list) -> list:
        to_close = []
        hard_stop_pct = self.config.get("hard_stop_loss_pct", 0.02)
        if self.initial_balance is None or self.initial_balance == 0:
            return to_close
        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pnl / self.initial_balance
            if pnl_pct <= -hard_stop_pct:
                logger.error(f"HARD STOP: {pos['symbol']} P&L {pnl_pct*100:.2f}%")
                to_close.append(pos)
        return to_close

    def open_position(self, symbol: str, action: str, price: float,
                      reason: str, drawdown_status: str) -> bool:
        volume = self.calculate_volume(price)
        if drawdown_status == "REDUCE":
            volume = max(1, volume // 2)
        side = "SIDE_BUY" if action == "BUY" else "SIDE_SELL"
        logger.info(f"Открытие {action} {volume} {symbol} @ {price:.4f}")
        result = self.arena.place_market_order(symbol, side, volume)
        if result:
            exec_price = result.get("exec_price", price)
            self.position_owners[symbol] = {
                "side": action,
                "entry_price": exec_price,
                "opened_at": datetime.now().isoformat()
            }
            self.save_positions()
            self.daily_orders += 1
            return True
        else:
            logger.error(f"Ордер не выполнен: {symbol}")
            return False

    def close_position(self, symbol: str, position: dict, reason: str) -> bool:
        side = "SIDE_SELL" if position['side'] == "BUY" else "SIDE_BUY"
        volume = int(position['quantity'])
        pnl = position.get('unrealized_pnl', 0)
        logger.info(f"Закрытие {position['side']} {volume} {symbol}: {reason}")
        result = self.arena.place_market_order(symbol, side, volume)
        if result:
            exec_price = result.get("exec_price", 0)
            commission = result.get("commission", 0)
            if symbol in self.position_owners:
                del self.position_owners[symbol]
                self.save_positions()
            self.daily_orders += 1
            return True
        else:
            logger.error(f"Не удалось закрыть {symbol}")
            return False
