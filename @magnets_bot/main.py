"""
Главный торговый бот для стратегии Минковского (акции MOEX)

Архитектура:
- StrategyManager: единая стратегия классификации Минковского
- Возвращает действия: BUY, SELL, CLOSE_LONG, CLOSE_SHORT или None
- Горячая перезагрузка config.json
- Риск-менеджмент: Hard Stop, Daily Limit, Drawdown Control

Файлы:
- config.json: настройки (перезагружается каждый цикл)
- state.json: состояние бота для UI
- trades.csv: лог сделок (append mode)
- positions_state.json: маппинг позиций к стратегии
- stop.flag: сигнал для остановки
"""

import json
import time
import logging
import os
from datetime import datetime
from typing import Optional, Dict
from arena_client import ArenaClient
from strategy import StrategyManager
from indicators import normalize_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.json"
TRADES_PATH = "trades.csv"
STATE_PATH = "state.json"
STOP_FLAG = "stop.flag"
POSITIONS_STATE = "positions_state.json"
STRATEGY_NAME = "MK"  # Имя стратегии для логов и CSV


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def log_trade(ticker: str, side: str, price: float, volume: int,
              reason: str = "", pnl: float = 0):
    """Запись сделки в CSV (append mode)"""
    file_exists = os.path.exists(TRADES_PATH)
    try:
        with open(TRADES_PATH, 'a', newline='', encoding='utf-8') as f:
            if not file_exists:
                f.write("timestamp,ticker,side,price,volume,strategy,reason,pnl\n")
            row = (
                f"{datetime.now().isoformat()},"
                f"{ticker},"
                f"{side},"
                f"{price:.4f},"
                f"{volume},"
                f"{STRATEGY_NAME},"
                f"\"{reason}\","
                f"{pnl:.2f}\n"
            )
            f.write(row)
    except Exception as e:
        logger.error(f"❌ Ошибка записи сделки: {e}")


def update_state(data: dict):
    """Обновление state.json для UI"""
    try:
        with open(STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"❌ Ошибка обновления state.json: {e}")


def load_positions_state() -> Dict:
    """Загрузка маппинга открытых позиций"""
    if os.path.exists(POSITIONS_STATE):
        try:
            with open(POSITIONS_STATE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_positions_state(data: Dict):
    """Сохранение маппинга открытых позиций"""
    try:
        with open(POSITIONS_STATE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения positions_state.json: {e}")


# ============================================================
# ТОРГОВЫЙ БОТ
# ============================================================

class TradingBot:
    """Главный торговый бот для стратегии Минковского"""
    
    def __init__(self):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Инициализация компонентов
        self.arena = ArenaClient(CONFIG_PATH)
        self.strategy_manager = StrategyManager(self.config)
        
        # Состояние
        self.initial_balance = None
        self.daily_orders = 0
        self.last_reset_date = None
        self.is_stopped = False
        
        # Маппинг: symbol -> {"side": str, "entry_price": float, "opened_at": str}
        self.position_owners = load_positions_state()
    
    def check_stop_flag(self) -> bool:
        """Проверка флага остановки"""
        if os.path.exists(STOP_FLAG):
            try:
                os.remove(STOP_FLAG)
            except Exception:
                pass
            return True
        return False
    
    def calculate_volume(self, price: float) -> int:
        """Расчёт объёма позиции"""
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
        """Проверка дневного лимита заявок"""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_orders = 0
            self.last_reset_date = today
            logger.info(f"🔄 Сброс счётчика заявок. Сегодня: {today}")
        
        max_orders = self.config.get("max_daily_orders", 190)
        if self.daily_orders >= max_orders:
            logger.warning(f"⛔ Дневной лимит достигнут: {self.daily_orders}/{max_orders}")
            return False
        return True
    
    def check_drawdown(self) -> str:
        """Проверка просадки по equity (не по available_cash)"""
        if self.initial_balance is None or self.initial_balance == 0:
            return "OK"
        
        equity = self.arena.get_equity()
        if equity == 0:
            return "OK"
        
        drawdown_pct = max(0, (self.initial_balance - equity) / self.initial_balance)
        
        stop_pct = self.config.get("drawdown_stop_pct", 0.10)
        reduce_pct = self.config.get("drawdown_reduce_pct", 0.05)
        
        if drawdown_pct >= stop_pct:
            self.is_stopped = True
            logger.error(f"⛔ ПРОСАДКА {drawdown_pct*100:.2f}% >= {stop_pct*100}%. ПОЛНАЯ ОСТАНОВКА!")
            return "STOP"
        elif drawdown_pct >= reduce_pct:
            logger.warning(f"⚠ Просадка {drawdown_pct*100:.2f}% >= {reduce_pct*100}%. Уменьшаем объём")
            return "REDUCE"
        
        return "OK"
    
    def check_hard_stop(self, positions: list) -> list:
        """Проверка Hard Stop Loss по открытым позициям"""
        to_close = []
        hard_stop_pct = self.config.get("hard_stop_loss_pct", 0.02)
        
        if self.initial_balance is None or self.initial_balance == 0:
            return to_close
        
        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pnl / self.initial_balance
            
            if pnl_pct <= -hard_stop_pct:
                logger.error(
                    f"🛑 HARD STOP LOSS: {pos['symbol']} P&L {pnl:.2f} "
                    f"({pnl_pct*100:.2f}% <= -{hard_stop_pct*100}%)"
                )
                to_close.append(pos)
        
        return to_close
    
    def _open_position(self, symbol: str, action: str, price: float,
                      reason: str, drawdown_status: str) -> bool:
        """
        Открытие позиции по сигналу MK
        
        Args:
            symbol: тикер (например, SBER@MISX)
            action: "BUY" или "SELL"
            price: текущая цена
            reason: причина входа
            drawdown_status: "OK", "REDUCE" или "STOP"
        """
        # Рассчитываем объём
        volume = self.calculate_volume(price)
        
        # Уменьшаем объём при просадке
        if drawdown_status == "REDUCE":
            volume = max(1, volume // 2)
            logger.info(f"⚠ Объём уменьшен до {volume} из-за просадки")
        
        # Определяем сторону ордера
        side = "SIDE_BUY" if action == "BUY" else "SIDE_SELL"
        
        # Отправляем ордер
        logger.info(f"🚀 [{STRATEGY_NAME}] Открытие {action} {volume} {symbol} @ {price:.4f}")
        result = self.arena.place_market_order(symbol, side, volume)
        
        if result:
            exec_price = result.get("exec_price", price)
            commission = result.get("commission", 0)
            
            log_trade(
                ticker=symbol,
                side=action,
                price=exec_price,
                volume=volume,
                reason=reason,
                pnl=0
            )
            
            # Запоминаем позицию
            self.position_owners[symbol] = {
                "side": action,
                "entry_price": exec_price,
                "opened_at": datetime.now().isoformat()
            }
            save_positions_state(self.position_owners)
            
            self.daily_orders += 1
            logger.info(
                f"✅ [{STRATEGY_NAME}] Ордер исполнен: {action} {volume} "
                f"@ {exec_price:.4f} (комиссия: {commission:.2f})"
            )
            return True
        else:
            logger.error(f"❌ [{STRATEGY_NAME}] Ордер не выполнен")
            return False
    
    def _close_position(self, symbol: str, position: Dict, reason: str) -> bool:
        """Закрытие позиции"""
        side = "SIDE_SELL" if position['side'] == "BUY" else "SIDE_BUY"
        volume = int(position['quantity'])
        pnl = position.get('unrealized_pnl', 0)
        
        logger.info(
            f"🔻 [{STRATEGY_NAME}] Закрытие {position['side']} {volume} {symbol}. "
            f"Причина: {reason}"
        )
        
        result = self.arena.place_market_order(symbol, side, volume)
        
        if result:
            exec_price = result.get("exec_price", 0)
            commission = result.get("commission", 0)
            
            log_trade(
                ticker=symbol,
                side=f"CLOSE_{side}",
                price=exec_price,
                volume=volume,
                reason=reason,
                pnl=pnl
            )
            
            # Удаляем запись о позиции
            if symbol in self.position_owners:
                del self.position_owners[symbol]
                save_positions_state(self.position_owners)
            
            self.daily_orders += 1
            
            emoji = "💰" if pnl >= 0 else "💸"
            logger.info(
                f"{emoji} [{STRATEGY_NAME}] Позиция закрыта. "
                f"P&L: {pnl:+.2f} ₽ (комиссия: {commission:.2f})"
            )
            return True
        else:
            logger.error(f"❌ [{STRATEGY_NAME}] Не удалось закрыть позицию")
            return False
    
    def run(self):
        """Главный цикл торговли"""
        logger.info("=" * 70)
        logger.info("🚀 ЗАПУСК ТОРГОВОГО БОТА: LORENTZIAN CLASSIFICATION")
        logger.info("=" * 70)
        
        # Получаем начальный equity (не available_cash — она проседает при покупках)
        self.initial_balance = self.arena.get_equity()
        logger.info(f"💰 Начальный equity: {self.initial_balance:,.2f} ₽")
        
        stocks = self.config.get("stocks", [])
        if not stocks:
            logger.error("❌ В config.json не настроен список stocks!")
            return
        
        timeframe = self.config.get("timeframe", "15m")
        bars_depth_days = self.config.get("bars_depth_days", 30)
        
        logger.info(f"📋 Инструментов: {len(stocks)} акций")
        logger.info(f"⏱ Таймфрейм: {timeframe}, глубина: {bars_depth_days} дней")
        logger.info(f"   {stocks[:5]}{'...' if len(stocks) > 5 else ''}")
        
        while not self.check_stop_flag():
            try:
                # 1. Горячая перезагрузка конфига
                try:
                    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                        self.config = json.load(f)
                    # Обновляем параметры strategy_manager
                    self.strategy_manager.classifier.config = self.config
                    # Перечитываем таймфрейм и глубину
                    timeframe = self.config.get("timeframe", "15m")
                    bars_depth_days = self.config.get("bars_depth_days", 30)
                except Exception as e:
                    logger.warning(f"⚠ Не удалось перезагрузить конфиг: {e}")
                
                # 2. Проверка глобальной остановки
                if self.is_stopped:
                    update_state({
                        "status": "STOPPED",
                        "balance": self.arena.get_balance(),
                        "last_update": datetime.now().isoformat(),
                        "last_signals": ["⛔ STOPPED by drawdown"],
                    })
                    time.sleep(60)
                    continue
                
                # 3. Проверка просадки
                drawdown_status = self.check_drawdown()
                if drawdown_status == "STOP":
                    balance = self.arena.get_balance()
                    update_state({
                        "status": "STOPPED",
                        "balance": balance,
                        "equity": balance,
                        "cash": balance,
                        "available_cash": balance,
                        "open_positions": len(self.arena.get_positions()),
                        "daily_orders": self.daily_orders,
                        "drawdown_pct": 0.10,
                        "last_signals": ["⛔ STOPPED by drawdown (10%)"],
                        "last_update": datetime.now().isoformat(),
                    })
                    time.sleep(60)
                    continue
                
                # 4. Проверка дневного лимита
                if not self.check_daily_limit():
                    balance = self.arena.get_balance()
                    update_state({
                        "status": "LIMIT_REACHED",
                        "balance": balance,
                        "daily_orders": self.daily_orders,
                        "last_signals": [f"⛔ Дневной лимит: {self.daily_orders}/{self.config.get('max_daily_orders', 190)}"],
                        "last_update": datetime.now().isoformat(),
                    })
                    time.sleep(60)
                    continue
                
                # 5. Получаем актуальный баланс и позиции
                balance = self.arena.get_balance()
                account_info = self.arena.get_account_info()
                equity = account_info.get("equity", balance) if account_info else balance
                cash = account_info.get("cash", balance) if account_info else balance
                available_cash = account_info.get("available_cash", balance) if account_info else balance
                positions = self.arena.get_positions()
                
                # 6. Hard Stop Loss
                hard_stop_positions = self.check_hard_stop(positions)
                for pos in hard_stop_positions:
                    self._close_position(
                        pos['symbol'], pos,
                        reason=f"Hard Stop Loss (P&L {pos.get('unrealized_pnl', 0):+.2f} ₽)"
                    )
                
                # Обновляем позиции после закрытия hard stop
                positions = self.arena.get_positions()
                
                # 7. Сканируем все инструменты
                signals_log = []
                errors_log = []
                stats = {"BUY": 0, "SELL": 0, "CLOSE_LONG": 0, "CLOSE_SHORT": 0}
                
                for symbol in stocks:
                    try:
                        # Загружаем свечи
                        df = self.arena.get_bars(symbol, timeframe=timeframe, days=bars_depth_days)
                        
                        if df is None or df.empty:
                            errors_log.append(f"{symbol}: нет данных")
                            continue
                        
                        # Нормализуем DataFrame
                        df = normalize_df(df)
                        
                        # Ищем открытую позицию по этому инструменту
                        position = next(
                            (p for p in positions if p.get('symbol') == symbol),
                            None
                        )
                        
                        # Оцениваем тикер (StrategManager сам решит: вход или выход)
                        signal = self.strategy_manager.evaluate_ticker(symbol, df)
                        
                        if not signal:
                            continue
                        
                        action = signal['action']
                        stats[action] = stats.get(action, 0) + 1
                        
                        # ============================================================
                        # Обработка действий
                        # ============================================================
                        
                        if action == "CLOSE_LONG":
                            # Есть BUY позиция — закрываем
                            if position and position['side'] == "BUY":
                                success = self._close_position(symbol, position, signal['reason'])
                                if success:
                                    signals_log.append(f"🔻 [{STRATEGY_NAME}] {symbol}: CLOSED BUY ({signal['reason']})")
                        
                        elif action == "CLOSE_SHORT":
                            # Есть SELL позиция — закрываем
                            if position and position['side'] == "SELL":
                                success = self._close_position(symbol, position, signal['reason'])
                                if success:
                                    signals_log.append(f"🔻 [{STRATEGY_NAME}] {symbol}: CLOSED SELL ({signal['reason']})")
                        
                        elif action == "BUY":
                            # Входим в BUY (если нет позиции)
                            if position is None:
                                current_price = float(df['close'].iloc[-1])
                                success = self._open_position(
                                    symbol, "BUY", current_price, signal['reason'], drawdown_status
                                )
                                if success:
                                    signals_log.append(
                                        f"✅ [{STRATEGY_NAME}] {symbol}: BUY @ {current_price:.2f} "
                                        f"(pred={signal['prediction']:.1f}, bars={signal['bars_held']})"
                                    )
                        
                        elif action == "SELL":
                            # Входим в SELL (если нет позиции)
                            if position is None:
                                current_price = float(df['close'].iloc[-1])
                                success = self._open_position(
                                    symbol, "SELL", current_price, signal['reason'], drawdown_status
                                )
                                if success:
                                    signals_log.append(
                                        f"✅ [{STRATEGY_NAME}] {symbol}: SELL @ {current_price:.2f} "
                                        f"(pred={signal['prediction']:.1f}, bars={signal['bars_held']})"
                                    )
                    
                    except Exception as e:
                        error_msg = f"Ошибка {symbol}: {e}"
                        logger.error(error_msg, exc_info=True)
                        errors_log.append(error_msg)
                
                # 8. Обновляем state.json для UI
                drawdown = (
                    (self.initial_balance - balance) / self.initial_balance
                    if self.initial_balance else 0
                )
                
                update_state({
                    "status": "RUNNING",
                    "balance": balance,
                    "equity": equity,
                    "cash": cash,
                    "available_cash": available_cash,
                    "open_positions": len(positions),
                    "daily_orders": self.daily_orders,
                    "drawdown_pct": drawdown,
                    "last_signals": signals_log[-20:] if signals_log else ["Нет сигналов"],
                    "last_errors": errors_log[-5:] if errors_log else [],
                    "stats": stats,
                    "stocks_count": len(stocks),
                    "positions_owners": self.position_owners,
                    "last_update": datetime.now().isoformat(),
                })
                
                logger.info(
                    f"💤 Цикл завершён. "
                    f"Баланс: {balance:,.2f} ₽ | "
                    f"Позиций: {len(positions)} | "
                    f"Сигналов: {len(signals_log)} | "
                    f"Ошибок: {len(errors_log)}"
                )
                if any(v > 0 for v in stats.values()):
                    logger.info(f"📊 Статистика действий: {stats}")
                
                # 9. Ждём 1 минуту перед следующим циклом
                for _ in range(60):
                    if self.check_stop_flag():
                        return
                    time.sleep(1)
            
            except Exception as e:
                logger.error(f"❌ Критическая ошибка цикла: {e}", exc_info=True)
                try:
                    balance = self.arena.get_balance()
                    update_state({
                        "status": "ERROR",
                        "balance": balance,
                        "last_update": datetime.now().isoformat(),
                        "last_errors": [f"Критическая ошибка: {str(e)}"],
                    })
                except Exception:
                    pass
                time.sleep(30)
        
        logger.info("⛔ Бот остановлен пользователем")


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()