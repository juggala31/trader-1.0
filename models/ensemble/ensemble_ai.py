# ensemble_ai.py - Simple Ensemble AI
import logging

class SimpleEnsembleAI:
    def __init__(self):
        self.logger = logging.getLogger("FTMO_AI")
        print("🤖 Simple Ensemble AI Created")
    
    def predict_signal(self, symbol, df, price):
        return {"action": "hold", "confidence": 0.5}
