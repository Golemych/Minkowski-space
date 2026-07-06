"""
Управление состоянием сигналов для каждого тикера.
Переносит предыдущий сигнал при нейтральном (nz), решает вход/выход.
"""

import logging
from typing import Dict, Optional
from market_state import MarketState

logger = logging.getLogger(__name__)


class StrategyStateManager:
    def __init__(self, config: dict):
        self.config = config
        self._states: Dict[str, MarketState] = {}

    def update_state(self, ticker: str, market_state: MarketState) -> Optional[str]:
        """
        Сохраняет MarketState для тикера и возвращает действие:
        BUY, SELL, CLOSE_LONG, CLOSE_SHORT, None
        """
        prev = self._states.get(ticker)
        self._states[ticker] = market_state

        action = None
        raw = market_state.raw_signal
        persisted = market_state.signal

        if raw == 0 and prev is not None:
            persisted = prev.signal

        if persisted == 1:
            action = "BUY"
        elif persisted == -1:
            action = "SELL"
        elif persisted == 0:
            if prev is not None and prev.signal == 1:
                action = "CLOSE_LONG"
            elif prev is not None and prev.signal == -1:
                action = "CLOSE_SHORT"
            else:
                return None

        kernel_exit_long = (
            market_state.kernel_bearish
            and not market_state.kernel_reversal_up
        )
        kernel_exit_short = (
            market_state.kernel_bullish
            and not market_state.kernel_reversal_down
        )

        if action == "BUY" and kernel_exit_long:
            action = "CLOSE_LONG"
        elif action == "SELL" and kernel_exit_short:
            action = "CLOSE_SHORT"

        return action

    def get_state(self, ticker: str) -> Optional[MarketState]:
        return self._states.get(ticker)

    def reset(self, ticker: Optional[str] = None):
        if ticker:
            self._states.pop(ticker, None)
        else:
            self._states.clear()
