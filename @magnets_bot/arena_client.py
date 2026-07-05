"""
Finam Arena API Client + MOEX ISS Data Provider
Единый клиент для работы с:
- Finam Arena API (торговля, баланс, ордера)
- MOEX ISS API (свечи через агрегацию 1m → нужный ТФ)

Добавлено: Инкрементальное кэширование свечей в локальные CSV файлы.
"""

import requests
import time
import json
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArenaClient:
    """Клиент для работы с Finam Arena API и MOEX ISS API"""
    
    # MOEX ISS API (для данных)
    MOEX_BASE_URL = "https://iss.moex.com/iss"
    
    # Папка для локального кэша свечей
    CACHE_DIR = "cache"
    
    # MOEX поддерживает интервалы: 1, 10, 60 (минуты), 24 (день), 7 (неделя), 31 (месяц)
    # 15m и 5m НЕ поддерживаются напрямую, поэтому агрегируем из 1m
    MOEX_SUPPORTED_INTERVALS = {1, 10, 60, 24, 7, 31}
    
    # Маппинг рынков MOEX
    MOEX_MARKETS = {
        # Акции MOEX
        "SBER":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        "SBERP": {"engine": "stock", "market": "shares", "board": "TQBR"},
        "GAZP":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        "ROSN":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        "LKOH":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        "VTBR":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        "GMKN":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        "ALRS":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        "AFLT":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        "MGNT":  {"engine": "stock", "market": "shares", "board": "TQBR"},
        
    }
    
    # Маппинг таймфреймов на минуты для агрегации
    TIMEFRAME_MINUTES = {
        "1m":  1,
        "5m":  5,
        "15m": 15,
        "30m": 30,
        "1h":  60,
        "2h":  120,
        "4h":  240,
        "1d":  1440,  # 24 * 60
    }
    
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.api_secret = config.get("api_secret", "")
        self.account_id = config.get("account_id", "")
        self.base_url = config.get("base_url", "https://arena.finam.ru/v1")
        
        self.session_token = None
        self.session_expires = 0
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "FinamArenaBot/1.0"
        })
        
        # Оперативный кэш (в памяти)
        self.data_cache = {}
        
        # Создаём папку для дискового кэша
        self._ensure_cache_dir()
    
    # ============================================================
    # ДИСКОВЫЙ КЭШ (CSV)
    # ============================================================
    
    def _ensure_cache_dir(self):
        """Создаёт папку кэша если её нет"""
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
            logger.info(f"📁 Создана папка кэша: {self.CACHE_DIR}")
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> str:
        """Путь к файлу кэша для инструмента"""
        clean = self._clean_symbol(symbol)
        return os.path.join(self.CACHE_DIR, f"{clean}_{timeframe}.csv")
    
    def _load_cached_bars(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Загрузка свечей из локального CSV кэша"""
        path = self._get_cache_path(symbol, timeframe)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col="datetime", parse_dates=True)
            return df
        except Exception as e:
            logger.warning(f"⚠ Ошибка чтения кэша {symbol}: {e}")
            return None
    
    def _save_cached_bars(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Сохранение свечей в локальный CSV кэш"""
        self._ensure_cache_dir()
        path = self._get_cache_path(symbol, timeframe)
        try:
            df.to_csv(path)
        except Exception as e:
            logger.warning(f"⚠ Ошибка сохранения кэша {symbol}: {e}")
    
    # ============================================================
    # АВТОРИЗАЦИЯ (Arena API)
    # ============================================================
    
    def _get_session(self) -> bool:
        if time.time() < self.session_expires - 60:
            return True
        
        if not self.api_secret:
            logger.error("❌ api_secret не задан в config.json")
            return False
        
        url = f"{self.base_url}/sessions"
        payload = {"secret": self.api_secret}
        
        try:
            resp = self.session.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.session_token = data.get("token")
                self.session_expires = time.time() + 15 * 60
                logger.info(f"✅ Сессия обновлена для счета {self.account_id}")
                return True
            else:
                logger.error(f"❌ Ошибка авторизации {resp.status_code}: {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"❌ Исключение при авторизации: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.session_token}"}
    
    # ============================================================
    # СЧЕТ И ПОЗИЦИИ (Arena API)
    # ============================================================
    
    def get_account_info(self) -> Dict:
        if not self._get_session():
            return {}
        
        url = f"{self.base_url}/accounts/{self.account_id}"
        try:
            resp = self.session.get(url, headers=self._get_headers(), timeout=10)
            if resp.status_code != 200:
                logger.error(f"❌ get_account_info {resp.status_code}: {resp.text[:200]}")
                return {}
            
            data = resp.json()
            return {
                "account_id": data.get("account_id"),
                "equity": float(data.get("equity", {}).get("value", 0)),
                "unrealized_profit": float(data.get("unrealized_profit", {}).get("value", 0)),
                "cash": float(data.get("cash", {}).get("value", 0)),
                "available_cash": float(data.get("available_cash", {}).get("value", 0)),
                "positions": data.get("positions", [])
            }
        except Exception as e:
            logger.error(f"❌ get_account_info: {e}")
            return {}
    
    def get_balance(self) -> float:
        info = self.get_account_info()
        return info.get("available_cash", 0)
    
    def get_equity(self) -> float:
        info = self.get_account_info()
        return info.get("equity", 0)
    
    def get_positions(self) -> List[Dict]:
        info = self.get_account_info()
        raw_positions = info.get("positions", [])
        
        positions = []
        for pos in raw_positions:
            try:
                qty = float(pos.get("quantity", {}).get("value", 0))
                avg_price = float(pos.get("average_price", {}).get("value", 0))
                pnl = float(pos.get("unrealized_pnl", {}).get("value", 0))
                side = "BUY" if qty > 0 else "SELL"
                qty = abs(qty)
                
                positions.append({
                    "symbol": pos.get("symbol"),
                    "quantity": qty,
                    "side": side,
                    "avg_price": avg_price,
                    "unrealized_pnl": pnl
                })
            except Exception as e:
                logger.error(f"⚠ Ошибка парсинга позиции: {e}")
        
        return positions
    
    # ============================================================
    # ТОРГОВЫЕ ОРДЕРА (Arena API)
    # ============================================================
    
    def place_market_order(self, symbol: str, side: str, quantity: int) -> Optional[Dict]:
        if not self._get_session():
            return None
        
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": {"value": str(int(quantity))},
            "type": "ORDER_TYPE_MARKET"
        }
        
        try:
            resp = self.session.post(url, json=payload, headers=self._get_headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                order = data.get("order", data)
                exec_price = float(order.get("execution_price", {}).get("value", 0))
                commission = float(order.get("commission", {}).get("value", 0))
                logger.info(f"🚀 Ордер: {side} {quantity} {symbol} @ {exec_price:.4f} (комиссия: {commission:.2f})")
                return {"order_id": data.get("order_id"), "exec_price": exec_price, "commission": commission}
            else:
                logger.error(f"❌ Ордер {resp.status_code}: {resp.text[:300]}")
                return None
        except Exception as e:
            logger.error(f"❌ place_market_order: {e}")
            return None
    
    # ============================================================
    # ИСТОРИЧЕСКИЕ ДАННЫЕ (MOEX ISS API + КЭШИРОВАНИЕ)
    # ============================================================
    
    def _clean_symbol(self, symbol: str) -> str:
        """Очистка: SBER@MISX → SBER"""
        clean = symbol.split('@')[0].split('.')[0].upper()
        return clean
    
    def _get_moex_market_info(self, symbol: str) -> Optional[Dict]:
        """Информация о рынке для инструмента"""
        clean_symbol = self._clean_symbol(symbol)
        info = self.MOEX_MARKETS.get(clean_symbol)
        if info:
            return info
        return {"engine": "stock", "market": "shares", "board": "TQBR"}
    
    def _fetch_1m_candles(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Загрузка 1-минутных свечей с MOEX ISS API.
        1m интервал поддерживает глубину до 30 дней.
        """
        clean_symbol = self._clean_symbol(symbol)
        market_info = self._get_moex_market_info(symbol)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = (
            f"{self.MOEX_BASE_URL}/engines/{market_info['engine']}/"
            f"markets/{market_info['market']}/boards/{market_info['board']}/"
            f"securities/{clean_symbol}/candles.json"
        )
        
        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "till": end_date.strftime("%Y-%m-%d"),
            "interval": 1,  # 1 минута
            "start": 0
        }
        
        logger.info(f"📥 MOEX: Загрузка {clean_symbol} (1m свечи для агрегации, {days} дней)...")
        
        try:
            all_candles = []
            start_page = 0
            
            while True:
                params["start"] = start_page
                resp = self.session.get(url, params=params, timeout=30)
                
                if resp.status_code != 200:
                    logger.error(f"❌ MOEX {resp.status_code}: {resp.text[:300]}")
                    return None
                
                data = resp.json()
                candles_data = data.get("candles", {})
                columns = candles_data.get("columns", [])
                values = candles_data.get("data", [])
                
                if not values:
                    break
                
                try:
                    open_idx = columns.index("open")
                    high_idx = columns.index("high")
                    low_idx = columns.index("low")
                    close_idx = columns.index("close")
                    volume_idx = columns.index("volume")
                    begin_idx = columns.index("begin")
                except ValueError as e:
                    logger.error(f"❌ Не найдена колонка: {e}")
                    return None
                
                for row in values:
                    try:
                        dt = datetime.strptime(row[begin_idx], "%Y-%m-%d %H:%M:%S")
                        all_candles.append({
                            "datetime": dt,
                            "open": float(row[open_idx]),
                            "high": float(row[high_idx]),
                            "low": float(row[low_idx]),
                            "close": float(row[close_idx]),
                            "volume": float(row[volume_idx])
                        })
                    except Exception:
                        continue
                
                # MOEX возвращает по 500 свечей на страницу
                if len(values) < 500 or start_page > 10000:
                    break
                
                start_page += 500
            
            if not all_candles:
                logger.warning(f"⚠ Пустой ответ для {clean_symbol}")
                return None
            
            df = pd.DataFrame(all_candles)
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='last')]
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки 1m для {clean_symbol}: {e}")
            return None
    
    def _aggregate_candles(self, df_1m: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
        """
        Агрегация 1-минутных свечей в нужный таймфрейм.
        """
        if df_1m.empty:
            return df_1m
        
        rule = f"{target_minutes}min"
        
        aggregated = df_1m.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        return aggregated
    
    def _fetch_and_aggregate(self, symbol: str, target_minutes: int, days: int) -> Optional[pd.DataFrame]:
        """Загрузка + агрегация (вынесено для переиспользования)"""
        if target_minutes in self.MOEX_SUPPORTED_INTERVALS:
            return self._fetch_candles_direct(symbol, target_minutes, days)
        else:
            df_1m = self._fetch_1m_candles(symbol, days)
            if df_1m is None:
                return None
            return self._aggregate_candles(df_1m, target_minutes)
    
    def get_bars(self, symbol: str, timeframe: str = "15m", 
                 days: int = 30) -> Optional[pd.DataFrame]:
        """
        Загрузка исторических свечей с ИНКРЕМЕНТАЛЬНЫМ КЭШИРОВАНИЕМ.
        
        Логика:
        1. Проверяем оперативный кэш (5 минут)
        2. Если есть дисковый кэш и он свежий — возвращаем из файла
        3. Если дисковый кэш устарел — догружаем только новые свечи
        4. Если кэша нет — полная загрузка + сохранение
        """
        cache_key = f"{symbol}_{timeframe}_{days}"
        
        # 1. Проверка оперативного кэша (5 минут)
        if cache_key in self.data_cache:
            cached_time, cached_data = self.data_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < 300:
                return cached_data
        
        target_minutes = self.TIMEFRAME_MINUTES.get(timeframe)
        if not target_minutes:
            logger.error(f"❌ Неизвестный таймфрейм: {timeframe}")
            return None
        
        now = datetime.now()
        
        # 2. Пытаемся загрузить из локального CSV кэша
        cached_df = self._load_cached_bars(symbol, timeframe)
        
        if cached_df is not None and len(cached_df) > 0:
            last_cached_date = cached_df.index[-1]
            
            # Если последняя свеча в кэше новее чем 5 минут назад — кэш актуален
            if (now - last_cached_date).total_seconds() < 300:
                cutoff = now - timedelta(days=days)
                result = cached_df[cached_df.index >= cutoff]
                self.data_cache[cache_key] = (now, result)
                return result
            
            # Кэш устарел — догружаем только новые свечи
            gap_days = max(1, int((now - last_cached_date).days) + 1)
            logger.info(f"📥 MOEX: Догрузка {self._clean_symbol(symbol)} ({gap_days} дней)...")
            
            new_df = self._fetch_and_aggregate(symbol, target_minutes, gap_days)
            
            if new_df is not None and len(new_df) > 0:
                # Объединяем старый кэш с новыми данными
                combined = pd.concat([cached_df, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                
                # Сохраняем обновлённый кэш
                self._save_cached_bars(symbol, timeframe, combined)
                
                # Обрезаем до нужной глубины
                cutoff = now - timedelta(days=days)
                result = combined[combined.index >= cutoff]
                
                self.data_cache[cache_key] = (now, result)
                logger.info(f"✅ Догружено {len(new_df)} новых свечей. Всего: {len(result)}")
                return result
            else:
                # Догрузка не удалась — возвращаем старый кэш как есть
                cutoff = now - timedelta(days=days)
                result = cached_df[cached_df.index >= cutoff]
                self.data_cache[cache_key] = (now, result)
                return result
        
        # 3. Кэша нет — полная загрузка
        fetch_days = min(days, 30)
        logger.info(f"📥 MOEX: Полная загрузка {self._clean_symbol(symbol)} ({fetch_days} дней)...")
        df = self._fetch_and_aggregate(symbol, target_minutes, fetch_days)
        
        if df is None or df.empty:
            logger.warning(f"⚠ Пустой результат для {symbol} ({timeframe})")
            return None
        
        # Сохраняем в локальный кэш
        self._save_cached_bars(symbol, timeframe, df)
        
        self.data_cache[cache_key] = (now, df)
        
        logger.info(f"✅ Загружено {len(df)} свечей {timeframe} для {symbol}")
        logger.info(f"   📅 {df.index[0]} → {df.index[-1]}")
        logger.info(f"   💹 Цена: {df['close'].iloc[-1]:.2f}")
        
        return df
    
    def _fetch_candles_direct(self, symbol: str, interval: int, days: int) -> Optional[pd.DataFrame]:
        """Прямая загрузка свечей с поддерживаемым MOEX интервалом"""
        clean_symbol = self._clean_symbol(symbol)
        market_info = self._get_moex_market_info(symbol)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = (
            f"{self.MOEX_BASE_URL}/engines/{market_info['engine']}/"
            f"markets/{market_info['market']}/boards/{market_info['board']}/"
            f"securities/{clean_symbol}/candles.json"
        )
        
        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "till": end_date.strftime("%Y-%m-%d"),
            "interval": interval,
            "start": 0
        }
        
        logger.info(f"📥 MOEX: Загрузка {clean_symbol} (interval={interval}, {days} дней)...")
        
        try:
            all_candles = []
            start_page = 0
            
            while True:
                params["start"] = start_page
                resp = self.session.get(url, params=params, timeout=30)
                
                if resp.status_code != 200:
                    logger.error(f"❌ MOEX {resp.status_code}")
                    return None
                
                data = resp.json()
                candles_data = data.get("candles", {})
                columns = candles_data.get("columns", [])
                values = candles_data.get("data", [])
                
                if not values:
                    break
                
                try:
                    open_idx = columns.index("open")
                    high_idx = columns.index("high")
                    low_idx = columns.index("low")
                    close_idx = columns.index("close")
                    volume_idx = columns.index("volume")
                    begin_idx = columns.index("begin")
                except ValueError as e:
                    logger.error(f"❌ Не найдена колонка: {e}")
                    return None
                
                for row in values:
                    try:
                        dt = datetime.strptime(row[begin_idx], "%Y-%m-%d %H:%M:%S")
                        all_candles.append({
                            "datetime": dt,
                            "open": float(row[open_idx]),
                            "high": float(row[high_idx]),
                            "low": float(row[low_idx]),
                            "close": float(row[close_idx]),
                            "volume": float(row[volume_idx])
                        })
                    except Exception:
                        continue
                
                if len(values) < 500 or start_page > 10000:
                    break
                
                start_page += 500
            
            if not all_candles:
                return None
            
            df = pd.DataFrame(all_candles)
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='last')]
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка прямой загрузки для {clean_symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str) -> Optional[float]:
        """Текущая цена через MOEX ISS"""
        clean_symbol = self._clean_symbol(symbol)
        market_info = self._get_moex_market_info(symbol)
        
        url = (
            f"{self.MOEX_BASE_URL}/engines/{market_info['engine']}/"
            f"markets/{market_info['market']}/boards/{market_info['board']}/"
            f"securities/{clean_symbol}.json"
        )
        
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            market_data = data.get("marketdata", {})
            columns = market_data.get("columns", [])
            values = market_data.get("data", [])
            
            if not values:
                return None
            
            try:
                last_idx = columns.index("LAST")
                price = values[0][last_idx]
                return float(price) if price is not None else None
            except (ValueError, IndexError):
                return None
                
        except Exception as e:
            logger.error(f"❌ get_quote: {e}")
            return None


# ==========================================
# ТЕСТ
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 ТЕСТ ARENA + MOEX ISS (С ИНКРЕМЕНТАЛЬНЫМ КЭШИРОВАНИЕМ)")
    print("=" * 70)
    
    client = ArenaClient()
    
    print("\n1️⃣  Баланс счета (Arena):")
    info = client.get_account_info()
    if info:
        print(f"   💰 Equity:        {info['equity']:.2f} ₽")
        print(f"   💵 Cash:          {info['cash']:.2f} ₽")
        print(f"   ✅ Available:     {info['available_cash']:.2f} ₽")
    else:
        print("   ❌ Не удалось")
    
    print("\n2️⃣  Свечи SBER@MISX (15m, 7 дней) — ПЕРВАЯ ЗАГРУЗКА:")
    df = client.get_bars("SBER@MISX", timeframe="15m", days=7)
    if df is not None:
        print(f"   ✅ {len(df)} свечей, последняя цена: {df['close'].iloc[-1]:.2f}")
        print(f"   📅 {df.index[0]} → {df.index[-1]}")
    else:
        print("   ❌ Не удалось")
    
    print("\n3️⃣  Свечи SBER@MISX (15m, 7 дней) — ИЗ КЭША:")
    df = client.get_bars("SBER@MISX", timeframe="15m", days=7)
    if df is not None:
        print(f"   ✅ {len(df)} свечей (из кэша), последняя цена: {df['close'].iloc[-1]:.2f}")
    else:
        print("   ❌ Не удалось")
    
    print("\n4️⃣  Текущая цена SBER@MISX:")
    quote = client.get_quote("SBER@MISX")
    if quote:
        print(f"   💹 Цена: {quote:.2f}")
    else:
        print("   ❌ Не удалось")
    
    print("\n" + "=" * 70)
    print("💡 Свечи кэшируются в папку cache/")
    print("   Повторные запросы выполняются мгновенно!")
    print("=" * 70)