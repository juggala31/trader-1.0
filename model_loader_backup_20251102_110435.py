import pickle
import os
from models.ensemble.professional_ensemble import ProfessionalEnsemble
from market_regime_detector import MarketRegimeDetector

class ModelLoader:
    def __init__(self, model_path="models/ensemble/trained_models/"):
        self.model_path = model_path
        self.ensemble = ProfessionalEnsemble()
        self.regime_detector = MarketRegimeDetector()
        self.models_loaded = False
    
    def load_trained_models(self):
        """Load models trained during weekend backtesting"""
        try:
            ensemble_path = os.path.join(self.model_path, "trained_ensemble.pkl")
            regime_path = os.path.join(self.model_path, "trained_regime_detector.pkl")
            
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'rb') as f:
                    self.ensemble = pickle.load(f)
                print("✓ Pre-trained ensemble models loaded")
            else:
                print("⚠️  No pre-trained ensemble found. Using default models.")
            
            if os.path.exists(regime_path):
                with open(regime_path, 'rb') as f:
                    self.regime_detector = pickle.load(f)
                print("✓ Pre-trained regime detector loaded")
            else:
                print("⚠️  No pre-trained regime detector found. Will train on live data.")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading trained models: {e}")
            return False
    
    def get_trained_ensemble(self):
        """Get the trained ensemble model"""
        return self.ensemble if self.models_loaded else ProfessionalEnsemble()
    
    def get_trained_regime_detector(self):
        """Get the trained regime detector"""
        return self.regime_detector if self.models_loaded else MarketRegimeDetector()

# Integration with your live trading system
def enhance_live_system_with_trained_models():
    """Example of how to integrate trained models into live system"""
    print("Enhancing live system with weekend-trained models...")
    
    model_loader = ModelLoader()
    if model_loader.load_trained_models():
        # Now you can use these pre-trained models in your live system
        trained_ensemble = model_loader.get_trained_ensemble()
        trained_regime_detector = model_loader.get_trained_regime_detector()
        
        print("🎯 Live system enhanced with weekend-trained models!")
        print("Your XGBoost models are now smarter from historical learning")
        
        return trained_ensemble, trained_regime_detector
    else:
        print("Using default models for this week")
        return ProfessionalEnsemble(), MarketRegimeDetector()

if __name__ == "__main__":
    enhance_live_system_with_trained_models()
