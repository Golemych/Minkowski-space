"""
Математическое ядро стратегии Минковского.
Stateless: только анализ рынка, без позиционной логики.
"""

import numpy as np
import pandas as pd
import logging
from indicators import Indicators, Normalizers, KernelRegression, Filters, normalize_df

logger = logging.getLogger(__name__)


class MinkowskiClassifier:
    DIRECTION_LONG = 1
    DIRECTION_SHORT = -1
    DIRECTION_NEUTRAL = 0

    def __init__(self, config: dict):
        self.config = config
        self.neighbors_count = config.get("neighbors_count", 8)
        self.max_bars_back = config.get("max_bars_back", 2000)
        self.feature_count = config.get("feature_count", 5)
        self.use_dynamic_exits = config.get("use_dynamic_exits", False)

        self.features = config.get("features", ["RSI", "WT", "CCI", "ADX", "RSI"])
        self.feature_params = config.get("feature_params",
            [(14, 1), (10, 11), (20, 1), (20, 2), (9, 1)])

        self.use_kernel_filter = config.get("use_kernel_filter", True)
        self.use_kernel_smoothing = config.get("use_kernel_smoothing", False)
        self.kernel_h = config.get("kernel_h", 8)
        self.kernel_r = config.get("kernel_r", 8.0)
        self.kernel_x = config.get("kernel_x", 25)
        self.kernel_lag = config.get("kernel_lag", 2)

        self.use_volatility_filter = config.get("use_volatility_filter", True)
        self.use_regime_filter = config.get("use_regime_filter", True)
        self.use_adx_filter = config.get("use_adx_filter", False)
        self.use_ema_filter = config.get("use_ema_filter", False)
        self.use_sma_filter = config.get("use_sma_filter", False)

        self.regime_threshold = config.get("regime_threshold", -0.1)
        self.adx_threshold = config.get("adx_threshold", 20)
        self.ema_period = config.get("ema_period", 200)
        self.sma_period = config.get("sma_period", 200)

    def _get_feature_series(self, df, feat_name, param_a, param_b):
        if feat_name == "RSI":
            return Normalizers.n_rsi(df['close'], param_a)
        elif feat_name == "WT":
            return Normalizers.n_wt(df, param_a, param_b)
        elif feat_name == "CCI":
            return Normalizers.n_cci(df, param_a)
        elif feat_name == "ADX":
            return Normalizers.n_adx(df, param_a)
        raise ValueError(f"Unknown feature: {feat_name}")

    def evaluate(self, df: pd.DataFrame) -> dict:
        df = normalize_df(df)
        result = {
            'signal': self.DIRECTION_NEUTRAL,
            'prediction': 0.0,
            'filters_passed': False,
            'is_bullish_rate': False,
            'is_bearish_rate': False,
            'is_bullish_smooth': False,
            'is_bearish_smooth': False,
            'is_bullish_change': False,
            'is_bearish_change': False,
            'is_bullish_cross_alert': False,
            'is_bearish_cross_alert': False,
            'is_ema_uptrend': True,
            'is_ema_downtrend': True,
            'is_sma_uptrend': True,
            'is_sma_downtrend': True,
        }

        feature_arrays = []
        for i in range(self.feature_count):
            feat_name = self.features[i]
            param_a, param_b = self.feature_params[i]
            series = self._get_feature_series(df, feat_name, param_a, param_b)
            feature_arrays.append(series.values)

        current_features = np.array([arr[-1] for arr in feature_arrays])

        close = df['close'].values
        labels = np.zeros(len(close))
        for i in range(4, len(close)):
            if close[i - 4] < close[i]:
                labels[i] = self.DIRECTION_SHORT
            elif close[i - 4] > close[i]:
                labels[i] = self.DIRECTION_LONG

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

        filter_vol = Filters.filter_volatility(df, enabled=self.use_volatility_filter)
        filter_reg = Filters.regime_filter(df, self.regime_threshold, self.kernel_h, self.kernel_r, self.use_regime_filter)
        filter_adx = Filters.filter_adx(df, 14, self.adx_threshold, self.use_adx_filter)
        filter_all = filter_vol and filter_reg and filter_adx
        result['filters_passed'] = filter_all
        result['vol_filter'] = filter_vol
        result['regime_filter'] = filter_reg
        result['adx_filter'] = filter_adx

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

        if prediction > 0 and filter_all:
            result['signal'] = self.DIRECTION_LONG
        elif prediction < 0 and filter_all:
            result['signal'] = self.DIRECTION_SHORT

        if self.use_kernel_filter or self.use_dynamic_exits:
            yhat1 = KernelRegression.rational_quadratic(df['close'], self.kernel_h, self.kernel_r, self.kernel_x)
            yhat2 = KernelRegression.gaussian(df['close'], max(3, self.kernel_h - self.kernel_lag), self.kernel_x)
            if len(yhat1.dropna()) >= 3:
                result['was_bullish_rate'] = yhat1.iloc[-3] < yhat1.iloc[-2]
                result['was_bearish_rate'] = yhat1.iloc[-3] > yhat1.iloc[-2]
                result['is_bullish_rate'] = yhat1.iloc[-2] < yhat1.iloc[-1]
                result['is_bearish_rate'] = yhat1.iloc[-2] > yhat1.iloc[-1]
                result['is_bullish_change'] = result['is_bullish_rate'] and result['was_bearish_rate']
                result['is_bearish_change'] = result['is_bearish_rate'] and result['was_bullish_rate']
                if len(yhat2.dropna()) >= 1:
                    result['is_bullish_smooth'] = yhat2.iloc[-1] >= yhat1.iloc[-1]
                    result['is_bearish_smooth'] = yhat2.iloc[-1] <= yhat1.iloc[-1]
                if len(yhat2.dropna()) >= 2 and len(yhat1.dropna()) >= 2:
                    yhat2_prev = yhat2.iloc[-2]
                    yhat2_curr = yhat2.iloc[-1]
                    yhat1_prev = yhat1.iloc[-2]
                    yhat1_curr = yhat1.iloc[-1]
                    result['is_bullish_cross_alert'] = (yhat2_prev < yhat1_prev) and (yhat2_curr > yhat1_curr)
                    result['is_bearish_cross_alert'] = (yhat2_prev > yhat1_prev) and (yhat2_curr < yhat1_curr)

        return result
