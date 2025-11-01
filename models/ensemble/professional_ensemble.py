# professional_ensemble.py - Multi-Model Ensemble System
import pandas as pd
import numpy as np
import logging
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

from .advanced_features import AdvancedFeatureEngineer

logger = logging.getLogger('FTMO_AI')

class ProfessionalEnsembleAI:
    """Professional multi-model ensemble AI for trading"""
    
    def __init__(self, config_path: str = None):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.models = {}
        self.model_weights = {}
        self.is_trained = False
        self.performance_history = deque(maxlen=1000)
        
        # Ensemble configuration
        self.config = {
            'min_confidence': 0.15,
            'ensemble_voting_threshold': 0.6,
            'max_positions': 3,
            'risk_adjustment': True,
            'model_weights': {
                'xgboost': 1.0,
                'lightgbm': 0.9, 
                'random_forest': 0.8
            }
        }
        
        if config_path:
            self.load_config(config_path)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ensemble models"""
        try:
            # XGBoost model
            import xgboost as xgb
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.model_weights['xgboost'] = self.config['model_weights']['xgboost']
            logger.info("✅ XGBoost model initialized")
        except ImportError:
            logger.warning("⚠️ XGBoost not available")
        
        try:
            # LightGBM model
            import lightgbm as lgb
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.model_weights['lightgbm'] = self.config['model_weights']['lightgbm']
            logger.info("✅ LightGBM model initialized")
        except ImportError:
            logger.warning("⚠️ LightGBM not available")
        
        try:
            # Random Forest model
            from sklearn.ensemble import RandomForestClassifier
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.model_weights['random_forest'] = self.config['model_weights']['random_forest']
            logger.info("✅ Random Forest model initialized")
        except ImportError:
            logger.warning("⚠️ scikit-learn not available")
        
        if not self.models:
            logger.error("❌ No ML models available - ensemble disabled")
    
    def predict_signal(self, symbol: str, df: pd.DataFrame, current_price: float) -> Dict:
        """Get professional ensemble prediction"""
        if not self.models:
            return self._get_fallback_signal()
        
        try:
            # Extract professional features
            features = self.feature_engineer.extract_professional_features(symbol, df, current_price)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Get predictions from all available models
            predictions = {}
            confidences = {}
            model_performances = {}
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
                    try:
                        proba = model.predict_proba(feature_vector)[0]
                        
                        # Handle different class configurations
                        if len(model.classes_) >= 3:
                            # Multi-class: [hold, buy, sell]
                            predicted_class = np.argmax(proba)
                            confidence = np.max(proba)
                            predictions[model_name] = predicted_class
                            confidences[model_name] = confidence
                        else:
                            # Binary classification
                            predicted_class = 1 if proba[1] > 0.5 else 0
                            confidence = proba[1] if predicted_class == 1 else proba[0]
                            predictions[model_name] = predicted_class
                            confidences[model_name] = confidence
                            
                        # Track model performance (simulated for now)
                        model_performances[model_name] = self._get_model_performance(model_name)
                        
                    except Exception as e:
                        logger.warning(f"Model {model_name} prediction error: {e}")
                        continue
            
            if not predictions:
                return self._get_fallback_signal()
            
            # Weighted ensemble voting
            final_decision, ensemble_confidence, voting_details = self._weighted_ensemble_vote(
                predictions, confidences, model_performances
            )
            
            # Risk-adjusted confidence
            adjusted_confidence = self._risk_adjust_confidence(ensemble_confidence, features)
            
            # Map to action
            action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
            action = action_map.get(final_decision, 'hold')
            
            return {
                'action': action,
                'confidence': adjusted_confidence,
                'raw_confidence': ensemble_confidence,
                'ensemble_votes': predictions,
                'ensemble_confidences': confidences,
                'model_weights': self.model_weights,
                'model_performances': model_performances,
                'voting_details': voting_details,
                'features_used': len(features),
                'market_regime': features.get('volatility_regime', 2),
                'timestamp': datetime.now().isoformat(),
                'features_sample': dict(list(features.items())[:5])  # First 5 features for display
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction error for {symbol}: {e}")
            return self._get_fallback_signal()
    
    def _weighted_ensemble_vote(self, predictions: Dict, confidences: Dict, performances: Dict) -> Tuple[int, float, Dict]:
        """Professional weighted ensemble voting"""
        if not predictions:
            return 0, 0.0, {}
        
        # Calculate weighted votes
        vote_counts = defaultdict(float)
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            confidence = confidences.get(model_name, 0.5)
            weight = self.model_weights.get(model_name, 1.0)
            performance = performances.get(model_name, 0.5)
            
            # Combined weight: base weight * confidence * performance
            combined_weight = weight * confidence * performance
            vote_counts[prediction] += combined_weight
            total_weight += combined_weight
        
        if total_weight == 0:
            return 0, 0.0, {}
        
        # Get winning prediction
        winning_prediction = max(vote_counts.items(), key=lambda x: x[1])
        normalized_confidence = winning_prediction[1] / total_weight
        
        # Voting details for analysis
        voting_details = {
            'total_weight': total_weight,
            'winning_vote_weight': winning_prediction[1],
            'vote_distribution': dict(vote_counts),
            'decision_threshold': self.config['ensemble_voting_threshold']
        }
        
        # Only return high-confidence decisions
        if normalized_confidence > self.config['ensemble_voting_threshold']:
            return winning_prediction[0], normalized_confidence, voting_details
        
        return 0, normalized_confidence, voting_details  # Default to hold
    
    def _risk_adjust_confidence(self, base_confidence: float, features: Dict) -> float:
        """Advanced risk-adjusted confidence calculation"""
        if not self.config['risk_adjustment']:
            return base_confidence
        
        adjusted_confidence = base_confidence
        
        # Reduce confidence in high volatility
        volatility = features.get('volatility_1h', 1.0)
        if volatility > 2.0:  # High volatility
            adjustment = max(0.5, 1.0 - (volatility - 2.0) / 10.0)
            adjusted_confidence *= adjustment
        
        # Adjust for market regime
        regime = features.get('volatility_regime', 2)
        if regime == 3:  # High volatility regime
            adjusted_confidence *= 0.8
        elif regime == 1:  # Low volatility regime
            adjusted_confidence *= 1.1
        
        # Adjust for price position extremes
        price_position = features.get('price_position', 0.5)
        if price_position < 0.2 or price_position > 0.8:  # Near support/resistance
            adjusted_confidence *= 0.9
        
        # Time-of-day adjustment
        if features.get('is_asia_session', 0) == 1:  # Asia session typically lower volatility
            adjusted_confidence *= 1.05
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _get_model_performance(self, model_name: str) -> float:
        """Get model performance score (placeholder - would use real historical performance)"""
        # Simulated performance based on model type
        performance_scores = {
            'xgboost': 0.75,
            'lightgbm': 0.72, 
            'random_forest': 0.68
        }
        return performance_scores.get(model_name, 0.65)
    
    def train_ensemble(self, training_data: List[Dict], validation_data: List[Dict] = None):
        """Train the ensemble models (placeholder implementation)"""
        logger.info("🎯 Training professional ensemble AI...")
        
        # This would be implemented with real training data
        # For now, mark as trained for demonstration
        self.is_trained = True
        logger.info("✅ Ensemble training completed (placeholder)")
    
    def save_ensemble(self, filepath: str):
        """Save ensemble models and configuration"""
        try:
            ensemble_data = {
                'models': {name: joblib.dump(model, f"{filepath}_{name}.joblib") for name, model in self.models.items()},
                'config': self.config,
                'feature_engineer': self.feature_engineer,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(ensemble_data, filepath)
            logger.info(f"✅ Ensemble saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving ensemble: {e}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble models and configuration"""
        try:
            ensemble_data = joblib.load(filepath)
            self.models = ensemble_data.get('models', {})
            self.config.update(ensemble_data.get('config', {}))
            logger.info(f"✅ Ensemble loaded from {filepath}")
            self.is_trained = True
        except Exception as e:
            logger.error(f"❌ Error loading ensemble: {e}")
    
    def _get_fallback_signal(self) -> Dict:
        """Fallback signal when ensemble is unavailable"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'raw_confidence': 0.0,
            'ensemble_votes': {},
            'ensemble_confidences': {},
            'model_weights': {},
            'model_performances': {},
            'voting_details': {},
            'features_used': 0,
            'market_regime': 2,
            'timestamp': datetime.now().isoformat(),
            'features_sample': {}
        }
    
    def load_config(self, config_path: str):
        """Load ensemble configuration"""
        try:
            import json
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            self.config.update(loaded_config)
            logger.info("✅ Ensemble configuration loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load ensemble config: {e}")

class EnsemblePerformanceTracker:
    """Track and analyze ensemble performance"""
    
    def __init__(self):
        self.trade_history = []
        self.model_performance = defaultdict(list)
    
    def record_trade(self, symbol: str, prediction: Dict, actual_result: float, execution_price: float):
        """Record trade performance for analysis"""
        trade_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual_result': actual_result,
            'execution_price': execution_price,
            'success': actual_result > 0
        }
        self.trade_history.append(trade_record)
        
        # Update model performance tracking
        for model_name, vote in prediction.get('ensemble_votes', {}).items():
            model_success = actual_result > 0
            self.model_performance[model_name].append(model_success)
    
    def get_model_performance_stats(self) -> Dict:
        """Get performance statistics for each model"""
        stats = {}
        for model_name, results in self.model_performance.items():
            if results:
                win_rate = sum(results) / len(results) * 100
                stats[model_name] = {
                    'total_trades': len(results),
                    'win_rate': win_rate,
                    'recent_performance': sum(results[-10:]) / min(10, len(results)) * 100 if len(results) >= 5 else 0.0
                }
        return stats
    
    def get_ensemble_performance(self) -> Dict:
        """Get overall ensemble performance"""
        if not self.trade_history:
            return {'total_trades': 0, 'overall_win_rate': 0.0}
        
        successful_trades = sum(1 for trade in self.trade_history if trade['success'])
        total_trades = len(self.trade_history)
        win_rate = successful_trades / total_trades * 100 if total_trades > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'overall_win_rate': win_rate,
            'recent_win_rate': sum(1 for trade in self.trade_history[-10:] if trade.get('success', False)) / min(10, len(self.trade_history)) * 100
        }
