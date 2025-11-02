# enhanced_gui_integration.py - FIXED VERSION
import tkinter as tk
from tkinter import ttk
import sys
import os
sys.path.append('.')

def enhance_existing_gui(main_gui_instance, ensemble_system):
    """Enhance existing FTMO GUI with professional ensemble displays"""
    try:
        from models.ensemble.ensemble_gui_integration import ProfessionalEnsembleDisplay, EnsembleControlPanel
        
        # Create ensemble display section
        ensemble_display = ProfessionalEnsembleDisplay(main_gui_instance, ensemble_system)
        
        # Create control panel
        control_panel = EnsembleControlPanel(main_gui_instance, ensemble_system)
        
        print("✅ Professional ensemble GUI integration completed")
        return {
            'display': ensemble_display,
            'controls': control_panel,
            'version': 'professional_gui_1.0'
        }
        
    except ImportError as e:
        print(f"⚠️ Professional GUI integration not available: {e}")
        return None

def update_gui_with_prediction(gui_components, symbol, prediction):
    """Update GUI with new ensemble prediction"""
    if gui_components and 'display' in gui_components:
        gui_components['display'].update_ensemble_display(symbol, prediction)

# Simple test function
def test_gui_integration():
    """Test the GUI integration"""
    try:
        # Create a simple test window
        root = tk.Tk()
        root.title("Ensemble GUI Integration Test")
        
        # Test ensemble display
        from models.ensemble.ensemble_gui_integration import ProfessionalEnsembleDisplay
        
        # Mock ensemble system for testing
        class MockEnsemble:
            def __init__(self):
                self.config = {'min_confidence': 0.15}
        
        ensemble_display = ProfessionalEnsembleDisplay(root, MockEnsemble())
        
        # Test prediction update
        test_prediction = {
            'action': 'buy',
            'confidence': 0.75,
            'ensemble_votes': {'xgboost': 1, 'lightgbm': 1, 'random_forest': 1},
            'ensemble_confidences': {'xgboost': 0.8, 'lightgbm': 0.7, 'random_forest': 0.75},
            'market_regime': 2,
            'features_used': 42,
            'features_sample': {'rsi_14': 45.5, 'volatility_1h': 1.2, 'price_position': 0.6}
        }
        
        ensemble_display.update_ensemble_display('US30Z25.sim', test_prediction)
        
        print("✅ GUI integration test passed")
        root.mainloop()
        
    except Exception as e:
        print(f"❌ GUI integration test failed: {e}")

if __name__ == "__main__":
    test_gui_integration()
