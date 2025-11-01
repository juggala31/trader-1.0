# integrate_professional_ensemble.py
import sys
import os
sys.path.append('.')

def integrate_professional_ensemble():
    """Integrate the professional ensemble AI system"""
    try:
        # Try to import the professional ensemble
        from models.ensemble.professional_ensemble import ProfessionalEnsembleAI
        from models.ensemble.ensemble_performance_tracker import EnsemblePerformanceTracker
        
        # Initialize professional ensemble
        ensemble_ai = ProfessionalEnsembleAI('models/ensemble/professional_config.json')
        performance_tracker = EnsemblePerformanceTracker()
        
        print("✅ Professional Ensemble AI initialized")
        print("   - Multi-model architecture (XGBoost, LightGBM, Random Forest)")
        print("   - Advanced feature engineering (40+ features)")
        print("   - Risk-adjusted confidence scoring")
        print("   - Performance tracking system")
        
        return {
            'ensemble': ensemble_ai,
            'tracker': performance_tracker,
            'version': 'professional_1.0'
        }
        
    except ImportError as e:
        print(f"⚠️ Professional ensemble not available: {e}")
        # Fallback to basic ensemble
        try:
            from models.ensemble.ensemble_ai import SimpleEnsembleAI
            ai = SimpleEnsembleAI()
            print("✅ Basic Ensemble AI initialized (fallback)")
            return {'ensemble': ai, 'version': 'basic_1.0'}
        except ImportError:
            print("❌ No ensemble AI available")
            return None

if __name__ == "__main__":
    result = integrate_professional_ensemble()
    if result:
        print(f"🎯 Ensemble AI integration successful - Version: {result['version']}")
    else:
        print("❌ Ensemble AI integration failed")
