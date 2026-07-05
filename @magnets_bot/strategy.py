"""
Стратегия классификации в пространстве Минковского для торгового бота.

Архитектура:
- MinkowskiClassifier: математическое ядро (работает с DataFrame).
- StrategyManager: управляет состоянием для каждого тикера отдельно 
  (так как в Pine Script индикатор работает на одном графике, а у нас 36 акций).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from indicators import Indicators, Normalizers, KernelRegression, Filters, normalize_df

logger = logging.getLogger(__name__)

# Максимальный размер истории для каждого тикера (защита от утечки памяти)
MAX_HISTORY_SIZE = 500


class MinkowskiClassifier:
    """
    Математическое ядро стратегии.
    Вычисляет сигнал для ТЕКУЩЕГО (последнего) бара в переданном DataFrame,
    используя предыдущие бары как историю.
    """
    
    DIRECTION_LONG = 1
    DIRECTION_SHORT = -1
    DIRECTION_NEUTRAL = 0
    
    def __init__(self, config: dict):
        self.config = config
        
        # Настройки ML
        self.neighbors_count = config.get("neighbors_count", 8)
        self.max_bars_back = config.get("max_bars_back", 2000)
        self.feature_count = config.get("feature_count", 5)
        self.use_dynamic_exits = config.get("use_dynamic_exits", False)
        
        # Признаки
        self.features = config.get("features", ["RSI", "WT", "CCI", "ADX", "RSI"])
        self.feature_params = config.get("feature_params", 
            [(14, 1), (10, 11), (20, 1), (20, 2), (9, 1)])
        
        # Kernel
        self.use_kernel_filter = config.get("use_kernel_filter", True)
        self.use_kernel_smoothing = config.get("use_kernel_smoothing", False)
        self.kernel_h = config.get("kernel_h", 8)
        self.kernel_r = config.get("kernel_r", 8.0)
        self.kernel_x = config.get("kernel_x", 25)
        self.kernel_lag = config.get("kernel_lag", 2)
        
        # Фильтры
        self.use_volatility_filter = config.get("use_volatility_filter", True)
        self.use_regime_filter = config.get("use_regime_filter", True)
        self.use_adx_filter = config.get("use_adx_filter", False)
        self.use_ema_filter = config.get("use_ema_filter", False)
        self.use_sma_filter = config.get("use_sma_filter", False)
        
        self.regime_threshold = config.get("regime_threshold", -0.1)
        self.adx_threshold = config.get("adx_threshold", 20)
        self.ema_period = config.get("ema_period", 200)
        self.sma_period = config.get("sma_period", 200)

    def _get_feature_series(self, df: pd.DataFrame, feat_name: str, param_a: int, param_b: int) -> pd.Series:
        if feat_name == "RSI":
            return Normalizers.n_rsi(df['close'], param_a)
        elif feat_name == "WT":
            return Normalizers.n_wt(df, param_a, param_b)
        elif feat_name == "CCI":
            return Normalizers.n_cci(df, param_a)
        elif feat_name == "ADX":
            return Normalizers.n_adx(df, param_a)
        raise ValueError(f"Unknown feature: {feat_name}")

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """
        Оценивает последний бар в DataFrame.
        Возвращает словарь с метриками, сигналом, kernel state и флагами входа/выхода.
        """
        df = normalize_df(df)
        
        result = {
            'signal': self.DIRECTION_NEUTRAL,
            'prediction': 0.0,
            'start_long': False,
            'start_short': False,
            'end_long': False,
            'end_short': False,
            'filters_passed': False,
            # 🆕 Kernel state для StrategyManager
            'is_bullish_rate': False,
            'is_bearish_rate': False,
            'is_bullish_smooth': False,
            'is_bearish_smooth': False,
            'is_bullish_change': False,
            'is_bearish_change': False,
            'is_bullish_cross_alert': False,
            'is_bearish_cross_alert': False,
            # 🆕 EMA/SMA state
            'is_ema_uptrend': True,
            'is_ema_downtrend': True,
            'is_sma_uptrend': True,
            'is_sma_downtrend': True,
        }
        
        
        # 1. Вычисление признаков для всего DataFrame
        feature_arrays = []
        for i in range(self.feature_count):
            feat_name = self.features[i]
            param_a, param_b = self.feature_params[i]
            series = self._get_feature_series(df, feat_name, param_a, param_b)
            feature_arrays.append(series.values)
            
        current_features = np.array([arr[-1] for arr in feature_arrays])
        
        # 2. Метки (y_train) - ТОЧНО ПО PINE SCRIPT: src[4] < src[0]
        close = df['close'].values
        labels = np.zeros(len(close))
        for i in range(4, len(close)):
            if close[i - 4] < close[i]:
                labels[i] = self.DIRECTION_SHORT
            elif close[i - 4] > close[i]:
                labels[i] = self.DIRECTION_LONG
                
        # 3. ANN поиск (метрика Минковского)
        # Алгоритм: выбираем bars с d >= lastDistance (РАЗНООБРАЗНЫЕ, не ближайшие)
        # lastDistance=-1.0 → первый бар всегда проходит
        # По мере заполнения lastDistance растёт до 75-го перцентиля
        # FIFO eviction — хронологическое разнообразие
        size_loop = min(self.max_bars_back - 1, len(close) - 5)
        predictions = []
        distances = []
        last_distance = -1.0
        
        for i in range(4, size_loop + 4):
            if labels[i] == 0:
                continue
            
            d = 0.0
            for f_idx in range(self.feature_count):
                d += np.log(1 + np.abs(current_features[f_idx] - feature_arrays[f_idx][i]))
                
            if d >= last_distance and (i % 4) != 0:
                last_distance = d
                distances.append(d)
                predictions.append(labels[i])
                
                if len(predictions) > self.neighbors_count:
                    percentile_idx = int(round(self.neighbors_count * 3 / 4))
                    last_distance = distances[min(percentile_idx, len(distances) - 1)]
                    
                    distances.pop(0)
                    predictions.pop(0)
                    
        prediction = float(np.sum(predictions)) if predictions else 0.0
        result['prediction'] = prediction
        
        # 4. Фильтры
        filter_vol = Filters.filter_volatility(df, enabled=self.use_volatility_filter)
        filter_reg = Filters.regime_filter(df, self.regime_threshold, self.kernel_h, self.kernel_r, self.use_regime_filter)
        filter_adx = Filters.filter_adx(df, 14, self.adx_threshold, self.use_adx_filter)
        filter_all = filter_vol and filter_reg and filter_adx
        result['filters_passed'] = filter_all
        
        # 5. 🆕 EMA/SMA фильтры (здесь, чтобы не дублировать)
        if self.use_ema_filter:
            ema = Indicators.ema(df['close'], self.ema_period)
            last_close = float(df['close'].iloc[-1])
            last_ema = float(ema.iloc[-1])
            result['is_ema_uptrend'] = last_close > last_ema
            result['is_ema_downtrend'] = last_close < last_ema
        
        if self.use_sma_filter:
            sma = Indicators.sma(df['close'], self.sma_period)
            last_close = float(df['close'].iloc[-1])
            last_sma = float(sma.iloc[-1])
            result['is_sma_uptrend'] = last_close > last_sma
            result['is_sma_downtrend'] = last_close < last_sma
        
        # 6. Базовый сигнал
        if prediction > 0 and filter_all:
            result['signal'] = self.DIRECTION_LONG
        elif prediction < 0 and filter_all:
            result['signal'] = self.DIRECTION_SHORT
        # else остаётся NEUTRAL
            
        # 7. 🆕 Kernel state (rates, smooth, changes) — возвращаем всё, что нужно
        if self.use_kernel_filter or self.use_dynamic_exits:
            yhat1 = KernelRegression.rational_quadratic(df['close'], self.kernel_h, self.kernel_r, self.kernel_x)
            yhat2 = KernelRegression.gaussian(df['close'], max(3, self.kernel_h - self.kernel_lag), self.kernel_x)
            
            if len(yhat1.dropna()) >= 3:
                # Rates of change (наклон kernel) — ТОЧНО по Pine Script
                result['was_bullish_rate'] = yhat1.iloc[-3] < yhat1.iloc[-2]
                result['was_bearish_rate'] = yhat1.iloc[-3] > yhat1.iloc[-2]
                result['is_bullish_rate'] = yhat1.iloc[-2] < yhat1.iloc[-1]
                result['is_bearish_rate'] = yhat1.iloc[-2] > yhat1.iloc[-1]
                
                # Changes (смена направления kernel)
                result['is_bullish_change'] = result['is_bullish_rate'] and result['was_bearish_rate']
                result['is_bearish_change'] = result['is_bearish_rate'] and result['was_bullish_rate']
                
                # Smooth (для use_kernel_smoothing)
                if len(yhat2.dropna()) >= 1:
                    result['is_bullish_smooth'] = yhat2.iloc[-1] >= yhat1.iloc[-1]
                    result['is_bearish_smooth'] = yhat2.iloc[-1] <= yhat1.iloc[-1]
                    
                # Crossover alerts (ta.crossunder/yhat2,yhat1)
                if len(yhat2.dropna()) >= 2 and len(yhat1.dropna()) >= 2:
                    yhat2_prev = yhat2.iloc[-2]
                    yhat2_curr = yhat2.iloc[-1]
                    yhat1_prev = yhat1.iloc[-2]
                    yhat1_curr = yhat1.iloc[-1]
                    result['is_bullish_cross_alert'] = (yhat2_prev < yhat1_prev) and (yhat2_curr > yhat1_curr)
                    result['is_bearish_cross_alert'] = (yhat2_prev > yhat1_prev) and (yhat2_curr < yhat1_curr)
                    
        return result


class StrategyManager:
    """
    Менеджер состояния для множества тикеров.
    В Pine Script индикатор работает на одном графике и хранит var-переменные.
    В Python мы торгуем 36 акций, поэтому для каждой нужен свой экземпляр состояния.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.classifier = MinkowskiClassifier(config)
        
        # Состояние по каждому тикеру
        self.ticker_states: Dict[str, Dict] = {}
        
    def _get_state(self, ticker: str) -> Dict:
        if ticker not in self.ticker_states:
            self.ticker_states[ticker] = {
                'last_signal': 0,
                'bars_held': 0,
                'last_entry_bar': -1,
                'last_kernel_change_bar': -1,
                'is_valid_long_exit_history': [],
                'is_valid_short_exit_history': [],
                'bar_index': 0,
                'start_long_history': [],
                'start_short_history': [],
                'alert_bullish_history': [],
                'alert_bearish_history': [],
                # 🆕 Для isLastSignalBuy/Sell
                'signal_history': [],
                'ema_uptrend_history': [],
                'ema_downtrend_history': [],
                'sma_uptrend_history': [],
                'sma_downtrend_history': [],
            }
        return self.ticker_states[ticker]

    def _trim_history(self, history_list: List, max_size: int = MAX_HISTORY_SIZE):
        """Обрезает историю до max_size элементов (защита от утечки памяти)"""
        if len(history_list) > max_size:
            del history_list[:len(history_list) - max_size]

    def _ta_barssince(self, history: List[bool]) -> int:
        for i in range(len(history) - 1, -1, -1):
            if history[i]:
                return len(history) - 1 - i
        return len(history)

    def evaluate_ticker(self, ticker: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Главный метод для бота. Возвращает действие: BUY, SELL, CLOSE_LONG, CLOSE_SHORT или None.
        """
        state = self._get_state(ticker)
        eval_result = self.classifier.evaluate(df)
        
        # ============================================================
        # 🆕 ИСПРАВЛЕНО: nz(signal[1]) — точно по Pine Script
        # Если prediction=0 ИЛИ фильтры не прошли (signal == NEUTRAL),
        # используем предыдущий сигнал. Иначе — новый.
        # ============================================================
        if eval_result['signal'] == self.classifier.DIRECTION_NEUTRAL:
            signal = state['last_signal']
        else:
            signal = eval_result['signal']
            
        # Bar-count logic
        if signal != state['last_signal']:
            state['bars_held'] = 0
        else:
            state['bars_held'] += 1
            
        is_held_4 = state['bars_held'] == 4
        is_held_less_4 = 0 < state['bars_held'] < 4
        is_diff_signal = signal != state['last_signal']
        
        # Берём EMA/SMA из eval_result (уже посчитано в evaluate())
        is_ema_up = eval_result['is_ema_uptrend']
        is_ema_down = eval_result['is_ema_downtrend']
        is_sma_up = eval_result['is_sma_uptrend']
        is_sma_down = eval_result['is_sma_downtrend']
            
        is_buy_signal = (signal == 1) and is_ema_up and is_sma_up
        is_sell_signal = (signal == -1) and is_ema_down and is_sma_down
        
        is_new_buy = is_buy_signal and is_diff_signal
        is_new_sell = is_sell_signal and is_diff_signal
        
        # ============================================================
        # 🆕 ИСПРАВЛЕНО: Kernel filter использует RATE, а не CHANGE
        # Точно по Pine Script:
        # isBullish = useKernelFilter ? (useKernelSmoothing ? isBullishSmooth : isBullishRate) : true
        # ============================================================
        is_bullish_change = eval_result.get('is_bullish_change', False)
        is_bearish_change = eval_result.get('is_bearish_change', False)
        
        if self.classifier.use_kernel_filter:
            if self.classifier.use_kernel_smoothing:
                is_bullish = eval_result.get('is_bullish_smooth', False)
                is_bearish = eval_result.get('is_bearish_smooth', False)
            else:
                is_bullish = eval_result.get('is_bullish_rate', False)
                is_bearish = eval_result.get('is_bearish_rate', False)
        else:
            is_bullish, is_bearish = True, True
            
        # Entries
        start_long = is_new_buy and is_bullish and is_ema_up and is_sma_up
        start_short = is_new_sell and is_bearish and is_ema_down and is_sma_down
        
        # Сохраняем в историю с обрезкой
        state['start_long_history'].append(start_long)
        state['start_short_history'].append(start_short)
        # alertBullish = useKernelSmoothing ? isBullishCrossAlert : isBullishChange
        if self.classifier.use_kernel_smoothing:
            alert_bullish = eval_result.get('is_bullish_cross_alert', False)
            alert_bearish = eval_result.get('is_bearish_cross_alert', False)
        else:
            alert_bullish = is_bullish_change
            alert_bearish = is_bearish_change
        state['alert_bullish_history'].append(alert_bullish)
        state['alert_bearish_history'].append(alert_bearish)
        state['signal_history'].append(signal)
        state['ema_uptrend_history'].append(is_ema_up)
        state['ema_downtrend_history'].append(is_ema_down)
        state['sma_uptrend_history'].append(is_sma_up)
        state['sma_downtrend_history'].append(is_sma_down)
        
        # Обрезаем историю
        self._trim_history(state['start_long_history'])
        self._trim_history(state['start_short_history'])
        self._trim_history(state['alert_bullish_history'])
        self._trim_history(state['alert_bearish_history'])
        self._trim_history(state['signal_history'])
        self._trim_history(state['ema_uptrend_history'])
        self._trim_history(state['ema_downtrend_history'])
        self._trim_history(state['sma_uptrend_history'])
        self._trim_history(state['sma_downtrend_history'])
        
        # Dynamic Exits Logic
        bars_since_green_entry = self._ta_barssince(state['start_long_history'])
        bars_since_red_entry = self._ta_barssince(state['start_short_history'])
        bars_since_green_exit = self._ta_barssince(state['alert_bearish_history'])
        bars_since_red_exit = self._ta_barssince(state['alert_bullish_history'])
        
        is_valid_long_exit = bars_since_green_exit > bars_since_green_entry
        is_valid_short_exit = bars_since_red_exit > bars_since_red_entry
        
        state['is_valid_long_exit_history'].append(is_valid_long_exit)
        state['is_valid_short_exit_history'].append(is_valid_short_exit)
        self._trim_history(state['is_valid_long_exit_history'])
        self._trim_history(state['is_valid_short_exit_history'])
        
        # [1] previous bar values
        prev_valid_long = state['is_valid_long_exit_history'][-2] if len(state['is_valid_long_exit_history']) >= 2 else False
        prev_valid_short = state['is_valid_short_exit_history'][-2] if len(state['is_valid_short_exit_history']) >= 2 else False
        
        end_long_dynamic = is_bearish_change and prev_valid_long
        end_short_dynamic = is_bullish_change and prev_valid_short
        
        # ============================================================
        # 🆕 Strict Exits с isLastSignalBuy/Sell (ТОЧНО по Pine Script)
        # endLongTradeStrict = ((isHeldFourBars and isLastSignalBuy) 
        #                       or (isHeldLessThanFourBars and isNewSellSignal and isLastSignalBuy)) 
        #                      and startLongTrade[4]
        # ============================================================
        
        # isLastSignalBuy = signal[4] == 1 and ema_uptrend[4] and sma_uptrend[4]
        if len(state['signal_history']) >= 5:
            is_last_signal_buy = (
                state['signal_history'][-5] == self.classifier.DIRECTION_LONG and
                state['ema_uptrend_history'][-5] and
                state['sma_uptrend_history'][-5]
            )
            is_last_signal_sell = (
                state['signal_history'][-5] == self.classifier.DIRECTION_SHORT and
                state['ema_downtrend_history'][-5] and
                state['sma_downtrend_history'][-5]
            )
        else:
            is_last_signal_buy = False
            is_last_signal_sell = False
        
        # startLongTrade[4]
        if len(state['start_long_history']) >= 5:
            start_long_trade_4 = state['start_long_history'][-5]
        else:
            start_long_trade_4 = False
        
        if len(state['start_short_history']) >= 5:
            start_short_trade_4 = state['start_short_history'][-5]
        else:
            start_short_trade_4 = False
        
        # Точная формула Pine Script
        end_long_strict = (
            ((is_held_4 and is_last_signal_buy) or 
             (is_held_less_4 and is_new_sell and is_last_signal_buy)) and 
            start_long_trade_4
        )
        
        end_short_strict = (
            ((is_held_4 and is_last_signal_sell) or 
             (is_held_less_4 and is_new_buy and is_last_signal_sell)) and 
            start_short_trade_4
        )
        
        # Выбор типа выхода
        is_dynamic_exit_valid = (
            not self.classifier.use_ema_filter and 
            not self.classifier.use_sma_filter and 
            not self.classifier.use_kernel_smoothing
        )
        
        if self.classifier.use_dynamic_exits and is_dynamic_exit_valid:
            end_long = end_long_dynamic
            end_short = end_short_dynamic
        else:
            end_long = end_long_strict
            end_short = end_short_strict
            
        # Обновляем состояние
        state['last_signal'] = signal
        state['bar_index'] += 1
        
        # Формируем действие для бота
        action = None
        
        # Приоритет выходов над входами
        if end_long:
            action = "CLOSE_LONG"
        elif end_short:
            action = "CLOSE_SHORT"
        elif start_long:
            action = "BUY"
        elif start_short:
            action = "SELL"
            
        if action:
            return {
                'action': action,
                'ticker': ticker,
                'prediction': eval_result['prediction'],
                'signal': signal,
                'bars_held': state['bars_held'],
                'reason': f"MK {action} (pred={eval_result['prediction']:.1f}, bars={state['bars_held']})"
            }
            
        return None