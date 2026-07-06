from dataclasses import dataclass


@dataclass
class MarketState:
    """Результат анализа SignalEngine — только рыночная информация, без позиционной логики."""
    raw_signal: int           # -1, 0, 1  (до nz-переноса)
    signal: int               # -1, 0, 1  (после nz-переноса предыдущего)
    prediction: float         # сумма KNN соседей
    filters_passed: bool

    vol_filter: bool
    regime_filter: bool
    adx_filter: bool
    ema_filter: bool
    sma_filter: bool

    kernel_bullish: bool
    kernel_bearish: bool
    kernel_reversal_up: bool   # смена с медвежьего на бычий
    kernel_reversal_down: bool # смена с бычьего на медвежий
