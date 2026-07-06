"""
Сбор метрик, логирование сделок, обновление state.json.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TRADES_PATH = "trades.csv"
STATE_PATH = "state.json"
STRATEGY_NAME = "MK"


class MetricsCollector:
    def __init__(self):
        self.errors: List[str] = []
        self.signals: List[str] = []
        self.stats: Dict[str, int] = {}

    def log_trade(self, ticker: str, side: str, price: float, volume: int,
                  reason: str = "", pnl: float = 0):
        file_exists = os.path.exists(TRADES_PATH)
        try:
            with open(TRADES_PATH, 'a', newline='', encoding='utf-8') as f:
                if not file_exists:
                    f.write("timestamp,ticker,side,price,volume,strategy,reason,pnl\n")
                row = (
                    f"{datetime.now().isoformat()},"
                    f"{ticker},{side},{price:.4f},{volume},"
                    f"{STRATEGY_NAME},\"{reason}\",{pnl:.2f}\n"
                )
                f.write(row)
        except Exception as e:
            logger.error(f"Ошибка записи сделки: {e}")

    def update_state(self, data: dict):
        try:
            with open(STATE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Ошибка обновления state.json: {e}")

    def add_signal(self, msg: str):
        self.signals.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)

    def reset_cycle(self):
        self.errors.clear()
        self.signals.clear()
        self.stats.clear()
