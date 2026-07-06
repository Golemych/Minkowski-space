"""
Торговый бот Минковского v2.
Архитектура:
- MinkowskiClassifier: stateless, только анализ рынка -> MarketState
- StrategyStateManager: состояние сигналов, nz-перенос, решение вход/выход
- TradeExecutor: исполнение сделок и риск-менеджмент
- MetricsCollector: логирование сделок и state.json
"""

import json
import time
import logging
import os
from datetime import datetime
from arena_client import ArenaClient
from strategy import MinkowskiClassifier
from market_state import MarketState
from state_manager import StrategyStateManager
from trade_manager import TradeExecutor
from metrics_collector import MetricsCollector
from indicators import normalize_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.json"
STOP_FLAG = "stop.flag"


class TradingBot:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.arena = ArenaClient(config_path)
        self.classifier = MinkowskiClassifier(self.config)
        self.state_manager = StrategyStateManager(self.config)
        self.executor = TradeExecutor(self.config, self.arena)
        self.metrics = MetricsCollector()
        self.is_stopped = False

    def check_stop_flag(self) -> bool:
        if os.path.exists(STOP_FLAG):
            try:
                os.remove(STOP_FLAG)
            except Exception:
                pass
            return True
        return False

    def _build_market_state(self, result: dict) -> MarketState:
        return MarketState(
            raw_signal=result['signal'],
            signal=result['signal'],
            prediction=result['prediction'],
            filters_passed=result['filters_passed'],
            vol_filter=result.get('vol_filter', False),
            regime_filter=result.get('regime_filter', False),
            adx_filter=result.get('adx_filter', False),
            ema_filter=result.get('is_ema_uptrend', False) or result.get('is_ema_downtrend', False),
            sma_filter=result.get('is_sma_uptrend', False) or result.get('is_sma_downtrend', False),
            kernel_bullish=result.get('is_bullish_rate', False),
            kernel_bearish=result.get('is_bearish_rate', False),
            kernel_reversal_up=result.get('is_bullish_cross_alert', False),
            kernel_reversal_down=result.get('is_bearish_cross_alert', False),
        )

    def run(self):
        logger.info("=" * 60)
        logger.info("ЗАПУСК MK BOT v2")
        logger.info("=" * 60)

        self.executor.initial_balance = self.arena.get_equity()
        logger.info(f"Начальный equity: {self.executor.initial_balance:,.2f}")

        stocks = self.config.get("stocks", [])
        timeframe = self.config.get("timeframe", "15m")
        bars_depth_days = self.config.get("bars_depth_days", 30)
        logger.info(f"Инструментов: {len(stocks)}, таймфрейм: {timeframe}")

        while not self.check_stop_flag():
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.classifier.config = self.config
                timeframe = self.config.get("timeframe", "15m")
                bars_depth_days = self.config.get("bars_depth_days", 30)

                if self.is_stopped:
                    logger.warning("Бот остановлен из-за просадки")
                    time.sleep(60)
                    continue

                drawdown_status = self.executor.check_drawdown()
                if drawdown_status == "STOP":
                    self.is_stopped = True
                    self.metrics.update_state({"status": "STOPPED", "last_update": datetime.now().isoformat()})
                    time.sleep(60)
                    continue

                if not self.executor.check_daily_limit():
                    self.metrics.update_state({"status": "LIMIT_REACHED", "last_update": datetime.now().isoformat()})
                    time.sleep(60)
                    continue

                balance = self.arena.get_balance()
                account_info = self.arena.get_account_info()
                equity = account_info.get("equity", balance) if account_info else balance
                cash = account_info.get("cash", balance) if account_info else balance
                available_cash = account_info.get("available_cash", balance) if account_info else balance
                positions = self.arena.get_positions()

                hard_stop_positions = self.executor.check_hard_stop(positions)
                for pos in hard_stop_positions:
                    self.executor.close_position(pos['symbol'], pos, reason="Hard Stop Loss")

                positions = self.arena.get_positions()
                self.metrics.reset_cycle()

                for symbol in stocks:
                    try:
                        df = self.arena.get_bars(symbol, timeframe=timeframe, days=bars_depth_days)
                        if df is None or df.empty:
                            self.metrics.add_error(f"{symbol}: нет данных")
                            continue

                        df = normalize_df(df)
                        position = next((p for p in positions if p.get('symbol') == symbol), None)

                        result = self.classifier.evaluate(df)
                        market_state = self._build_market_state(result)

                        action = self.state_manager.update_state(symbol, market_state)

                        if not action:
                            continue

                        self.metrics.stats[action] = self.metrics.stats.get(action, 0) + 1

                        if action == "CLOSE_LONG" and position and position['side'] == "BUY":
                            success = self.executor.close_position(symbol, position, reason="Signal exit")
                            if success:
                                current_price = float(df['close'].iloc[-1])
                                self.metrics.log_trade(symbol, "CLOSE_LONG", current_price, int(position['quantity']), reason="Signal")
                                self.metrics.add_signal(f"CLOSE_LONG {symbol}")

                        elif action == "CLOSE_SHORT" and position and position['side'] == "SELL":
                            success = self.executor.close_position(symbol, position, reason="Signal exit")
                            if success:
                                current_price = float(df['close'].iloc[-1])
                                self.metrics.log_trade(symbol, "CLOSE_SHORT", current_price, int(position['quantity']), reason="Signal")
                                self.metrics.add_signal(f"CLOSE_SHORT {symbol}")

                        elif action == "BUY" and position is None:
                            current_price = float(df['close'].iloc[-1])
                            success = self.executor.open_position(symbol, "BUY", current_price, "Signal", drawdown_status)
                            if success:
                                self.metrics.log_trade(symbol, "BUY", current_price, self.executor.calculate_volume(current_price), reason="Signal")
                                self.metrics.add_signal(f"BUY {symbol} @ {current_price:.2f}")

                        elif action == "SELL" and position is None:
                            current_price = float(df['close'].iloc[-1])
                            success = self.executor.open_position(symbol, "SELL", current_price, "Signal", drawdown_status)
                            if success:
                                self.metrics.log_trade(symbol, "SELL", current_price, self.executor.calculate_volume(current_price), reason="Signal")
                                self.metrics.add_signal(f"SELL {symbol} @ {current_price:.2f}")

                    except Exception as e:
                        logger.error(f"Ошибка {symbol}: {e}", exc_info=True)
                        self.metrics.add_error(f"{symbol}: {e}")

                drawdown_pct = (
                    (self.executor.initial_balance - balance) / self.executor.initial_balance
                    if self.executor.initial_balance else 0
                )

                self.metrics.update_state({
                    "status": "RUNNING",
                    "balance": balance,
                    "equity": equity,
                    "cash": cash,
                    "available_cash": available_cash,
                    "open_positions": len(positions),
                    "daily_orders": self.executor.daily_orders,
                    "drawdown_pct": drawdown_pct,
                    "last_signals": self.metrics.signals[-20:] or ["Нет сигналов"],
                    "last_errors": self.metrics.errors[-5:] or [],
                    "stats": self.metrics.stats,
                    "stocks_count": len(stocks),
                    "positions_owners": self.executor.position_owners,
                    "last_update": datetime.now().isoformat(),
                })

                logger.info(
                    f"Цикл: баланс {balance:,.2f} | "
                    f"позиций {len(positions)} | "
                    f"сигналов {len(self.metrics.signals)} | "
                    f"ошибок {len(self.metrics.errors)}"
                )
                if any(v > 0 for v in self.metrics.stats.values()):
                    logger.info(f"Статистика: {self.metrics.stats}")

                for _ in range(60):
                    if self.check_stop_flag():
                        return
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Критическая ошибка: {e}", exc_info=True)
                try:
                    self.metrics.update_state({
                        "status": "ERROR",
                        "last_update": datetime.now().isoformat(),
                        "last_errors": [str(e)],
                    })
                except Exception:
                    pass
                time.sleep(30)

        logger.info("Бот остановлен")


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
