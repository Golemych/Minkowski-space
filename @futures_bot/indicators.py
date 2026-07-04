"""
Indicators: Bollinger Bands, Keltner Channel, Contango Calculator/Filter.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    return df


class Indicators:

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, length: int = 230,
                       deviation: float = 2.1) -> Tuple[pd.Series, pd.Series, pd.Series]:
        df = normalize_df(df)
        if len(df) < length:
            nan_series = pd.Series([float('nan')] * len(df), index=df.index)
            return nan_series, nan_series, nan_series
        middle = df['close'].rolling(window=length).mean()
        std = df['close'].rolling(window=length).std()
        upper = middle + (deviation * std)
        lower = middle - (deviation * std)
        return upper, middle, lower

    @staticmethod
    def keltner_channel(df: pd.DataFrame, ema_length: int = 150,
                       atr_length: int = 24, deviation: float = 3.9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        df = normalize_df(df)
        min_len = max(ema_length, atr_length)
        if len(df) < min_len:
            nan_series = pd.Series([float('nan')] * len(df), index=df.index)
            return nan_series, nan_series, nan_series
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        middle = typical_price.ewm(span=ema_length, adjust=False).mean()
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(atr_length).mean()
        upper = middle + (deviation * atr)
        lower = middle - (deviation * atr)
        return upper, middle, lower


class ContangoCalculator:

    @staticmethod
    def calculate(spot_price: float, futures_price: float, coeff: float) -> float:
        if spot_price == 0 or coeff == 0:
            return 0.0
        contango_abs = (futures_price / coeff) - spot_price
        contango_pct = contango_abs / (spot_price / 100)
        return contango_pct


class ContangoFilter:

    def __init__(self):
        self.contango_values = {}
        self.last_update = {}

    def update(self, ticker: str, contango_pct: float):
        self.contango_values[ticker] = contango_pct
        self.last_update[ticker] = datetime.now()

    def get_stage(self, ticker: str, top_count: int = 5) -> int:
        if ticker not in self.contango_values:
            return 0
        sorted_tickers = sorted(self.contango_values.items(), key=lambda x: x[1])
        total = len(sorted_tickers)
        if total == 0:
            return 0
        for i, (t, _) in enumerate(sorted_tickers):
            if t == ticker:
                if i < top_count:
                    return 1
                elif i >= total - top_count:
                    return 2
                else:
                    return 0
        return 0

    def get_ranking_table(self) -> str:
        if not self.contango_values:
            return "Нет данных"
        sorted_tickers = sorted(self.contango_values.items(), key=lambda x: x[1])
        total = len(sorted_tickers)
        lines = ["Ранжирование по контанго:", "-" * 50]
        for i, (ticker, pct) in enumerate(sorted_tickers):
            if i < 5:
                stage = "LONG"
            elif i >= total - 5:
                stage = "SHORT"
            else:
                stage = "---"
            lines.append(f"  {i+1:>2}. {ticker:<8} {pct:>8.3f}%  {stage}")
        lines.append("-" * 50)
        return "\n".join(lines)

    def get_all_stages(self, top_count: int = 5) -> dict:
        return {
            ticker: self.get_stage(ticker, top_count)
            for ticker in self.contango_values.keys()
        }
