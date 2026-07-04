import pandas as pd
from typing import Dict, List

class MinkowskiClassifier:
    def predict(self, data: pd.DataFrame) -> str:
        # Placeholder for prediction logic
        if data['close'].iloc[-1] > data['open'].iloc[-1]:
            return 'BUY'
        else:
            return 'SELL'

class StrategyManager:
    def __init__(self, classifier: MinkowskiClassifier):
        self.classifier = classifier

    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        signals = {}
        for symbol, df in data_dict.items():
            signal = self.classifier.predict(df)
            signals[symbol] = signal
        return signals