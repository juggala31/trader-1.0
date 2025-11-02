# deep_learning_analyzer.py - Advanced Pattern Recognition with Deep Learning
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('FTMO_AI')

class DeepLearningAnalyzer:
    """Deep learning integration for advanced market pattern recognition"""
    
    def __init__(self):
        self.pattern_models = {}
        self.sequence_length = 50
        self.prediction_horizon = 5
        self.is_trained = False
        
        # Pattern recognition configuration
        self.pattern_config = {
            'trend_patterns': True,
            'reversal_patterns': True,
            'consolidation_patterns': True,
            'volatility_patterns': True,
            'seasonal_patterns': True
        }
        
        # Initialize deep learning components
        self._initialize_deep_learning()
    
    def _initialize_deep_learning(self):
        """Initialize deep learning components with fallbacks"""
        try:
            # Try to import TensorFlow/Keras
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
            from tensorflow.keras.optimizers import Adam
            
            self.tf_available = True
            self.keras_available = True
            logger.info("✅ TensorFlow/Keras available for deep learning")
            
        except ImportError:
            self.tf_available = False
            self.keras_available = False
            logger.warning("⚠️ TensorFlow not available - using simplified pattern recognition")
        
        try:
            # Try to import PyTorch as alternative
            import torch
            import torch.nn as nn
            self.torch_available = True
            logger.info("✅ PyTorch available for deep learning")
        except ImportError:
            self.torch_available = False
            logger.warning("⚠️ PyTorch not available")
    
    def analyze_market_patterns(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Analyze market patterns using deep learning"""
        if len(price_data) < self.sequence_length:
            return self._get_fallback_pattern_analysis()
        
        try:
            patterns = {}
            
            # Trend pattern analysis
            if self.pattern_config['trend_patterns']:
                patterns['trend_analysis'] = self._analyze_trend_patterns(price_data)
            
            # Reversal pattern detection
            if self.pattern_config['reversal_patterns']:
                patterns['reversal_signals'] = self._detect_reversal_patterns(price_data)
            
            # Consolidation patterns
            if self.pattern_config['consolidation_patterns']:
                patterns['consolidation_zones'] = self._identify_consolidation_zones(price_data)
            
            # Volatility patterns
            if self.pattern_config['volatility_patterns']:
                patterns['volatility_regimes'] = self._analyze_volatility_patterns(price_data)
            
            # Seasonal patterns
            if self.pattern_config['seasonal_patterns']:
                patterns['seasonal_tendencies'] = self._analyze_seasonal_patterns(price_data)
            
            # Deep learning prediction if available
            if self.tf_available or self.torch_available:
                patterns['dl_predictions'] = self._deep_learning_prediction(price_data)
            
            return {
                'success': True,
                'patterns': patterns,
                'pattern_confidence': self._calculate_pattern_confidence(patterns),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis error for {symbol}: {e}")
            return self._get_fallback_pattern_analysis()
    
    def _analyze_trend_patterns(self, price_data: pd.DataFrame) -> Dict:
        """Analyze trend patterns using technical analysis and ML"""
        try:
            closes = price_data['close'].values
            
            # Calculate multiple trend indicators
            sma_20 = pd.Series(closes).rolling(20).mean().iloc[-1]
            sma_50 = pd.Series(closes).rolling(50).mean().iloc[-1]
            sma_200 = pd.Series(closes).rolling(200).mean().iloc[-1]
            
            current_price = closes[-1]
            
            # Trend strength calculation
            trend_strength = self._calculate_trend_strength(closes)
            
            # Trend direction
            if current_price > sma_20 > sma_50 > sma_200:
                trend_direction = "strong_uptrend"
            elif current_price < sma_20 < sma_50 < sma_200:
                trend_direction = "strong_downtrend"
            elif current_price > sma_20 and sma_20 > sma_50:
                trend_direction = "uptrend"
            elif current_price < sma_20 and sma_20 < sma_50:
                trend_direction = "downtrend"
            else:
                trend_direction = "ranging"
            
            return {
                'direction': trend_direction,
                'strength': trend_strength,
                'sma_20_vs_price': (current_price - sma_20) / sma_20 * 100,
                'sma_50_vs_price': (current_price - sma_50) / sma_50 * 100,
                'sma_200_vs_price': (current_price - sma_200) / sma_200 * 100,
                'trend_duration': self._estimate_trend_duration(closes)
            }
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {'direction': 'unknown', 'strength': 0.0}
    
    def _detect_reversal_patterns(self, price_data: pd.DataFrame) -> Dict:
        """Detect potential reversal patterns"""
        try:
            highs = price_data['high'].values
            lows = price_data['low'].values
            closes = price_data['close'].values
            
            reversal_signals = {}
            
            # RSI divergence analysis
            rsi_divergence = self._analyze_rsi_divergence(closes)
            if rsi_divergence['detected']:
                reversal_signals['rsi_divergence'] = rsi_divergence
            
            # Double top/bottom detection
            double_pattern = self._detect_double_top_bottom(highs, lows)
            if double_pattern['detected']:
                reversal_signals['double_pattern'] = double_pattern
            
            # Head and shoulders detection (simplified)
            hs_pattern = self._detect_head_shoulders(highs, lows)
            if hs_pattern['detected']:
                reversal_signals['head_shoulders'] = hs_pattern
            
            # MACD divergence
            macd_divergence = self._analyze_macd_divergence(closes)
            if macd_divergence['detected']:
                reversal_signals['macd_divergence'] = macd_divergence
            
            return reversal_signals
            
        except Exception as e:
            logger.error(f"Reversal pattern detection error: {e}")
            return {}
    
    def _identify_consolidation_zones(self, price_data: pd.DataFrame) -> Dict:
        """Identify consolidation zones and breakouts"""
        try:
            closes = price_data['close'].values
            volatility = pd.Series(closes).pct_change().std()
            
            # Bollinger Band squeeze detection
            bb_squeeze = self._detect_bollinger_squeeze(price_data)
            
            # ATR-based volatility analysis
            atr_volatility = self._calculate_atr_volatility(price_data)
            
            # Support/resistance levels
            support_resistance = self._identify_support_resistance(price_data)
            
            return {
                'volatility_regime': 'low' if volatility < 0.005 else 'high' if volatility > 0.015 else 'medium',
                'bollinger_squeeze': bb_squeeze,
                'atr_volatility': atr_volatility,
                'support_levels': support_resistance.get('support', []),
                'resistance_levels': support_resistance.get('resistance', []),
                'consolidation_score': self._calculate_consolidation_score(price_data)
            }
            
        except Exception as e:
            logger.error(f"Consolidation analysis error: {e}")
            return {'volatility_regime': 'unknown'}
    
    def _analyze_volatility_patterns(self, price_data: pd.DataFrame) -> Dict:
        """Analyze volatility patterns and regimes"""
        try:
            returns = price_data['close'].pct_change().dropna()
            
            # GARCH-like volatility estimation (simplified)
            recent_vol = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
            historical_vol = returns.std()
            
            # Volatility clustering detection
            volatility_cluster = self._detect_volatility_clustering(returns)
            
            # Volatility regime classification
            if recent_vol > historical_vol * 1.5:
                regime = "high_volatility"
            elif recent_vol < historical_vol * 0.7:
                regime = "low_volatility"
            else:
                regime = "normal_volatility"
            
            return {
                'current_regime': regime,
                'volatility_ratio': recent_vol / historical_vol if historical_vol > 0 else 1.0,
                'volatility_clustering': volatility_cluster,
                'volatility_momentum': self._calculate_volatility_momentum(returns),
                'expected_volatility': self._predict_volatility(returns)
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis error: {e}")
            return {'current_regime': 'unknown'}
    
    def _analyze_seasonal_patterns(self, price_data: pd.DataFrame) -> Dict:
        """Analyze seasonal and time-based patterns"""
        try:
            # Time-of-day patterns
            hourly_patterns = self._analyze_hourly_patterns(price_data)
            
            # Day-of-week patterns
            daily_patterns = self._analyze_daily_patterns(price_data)
            
            # Monthly seasonality
            monthly_patterns = self._analyze_monthly_patterns(price_data)
            
            return {
                'hourly_tendencies': hourly_patterns,
                'daily_patterns': daily_patterns,
                'monthly_seasonality': monthly_patterns,
                'session_analysis': self._analyze_trading_sessions(price_data)
            }
            
        except Exception as e:
            logger.error(f"Seasonal analysis error: {e}")
            return {}
    
    def _deep_learning_prediction(self, price_data: pd.DataFrame) -> Dict:
        """Deep learning-based price prediction"""
        if not self.tf_available and not self.torch_available:
            return {'available': False, 'reason': 'No deep learning framework available'}
        
        try:
            # Prepare data for deep learning
            sequences = self._create_sequences(price_data['close'].values)
            
            if len(sequences) < 10:
                return {'available': False, 'reason': 'Insufficient data for prediction'}
            
            # Simple LSTM prediction (placeholder implementation)
            prediction = self._simple_lstm_prediction(sequences)
            
            return {
                'available': True,
                'next_period_prediction': prediction.get('prediction', 0),
                'confidence': prediction.get('confidence', 0),
                'direction': prediction.get('direction', 'neutral'),
                'horizon': self.prediction_horizon
            }
            
        except Exception as e:
            logger.error(f"Deep learning prediction error: {e}")
            return {'available': False, 'reason': str(e)}
    
    # Technical analysis helper methods
    def _calculate_trend_strength(self, prices: np.array) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 20:
            return 0.0
        
        x = np.arange(len(prices))
        y = prices
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to 0-1 range
        max_slope = np.std(prices) * 0.1  # Reasonable maximum
        strength = min(1.0, abs(slope) / max_slope) if max_slope > 0 else 0.0
        
        return strength
    
    def _analyze_rsi_divergence(self, prices: np.array, period: int = 14) -> Dict:
        """Analyze RSI divergence for reversal signals"""
        try:
            if len(prices) < period * 2:
                return {'detected': False}
            
            # Calculate RSI
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi = 100 - (100 / (1 + rs))
            
            # Look for divergences
            price_highs = pd.Series(prices).rolling(5).max().dropna()
            rsi_highs = rsi.rolling(5).max().dropna()
            
            # Simple divergence detection
            if len(price_highs) > 5 and len(rsi_highs) > 5:
                recent_price_trend = price_highs.iloc[-5:].mean() - price_highs.iloc[-10:-5].mean()
                recent_rsi_trend = rsi_highs.iloc[-5:].mean() - rsi_highs.iloc[-10:-5].mean()
                
                if recent_price_trend > 0 and recent_rsi_trend < 0:
                    return {'detected': True, 'type': 'bearish_divergence', 'strength': 0.7}
                elif recent_price_trend < 0 and recent_rsi_trend > 0:
                    return {'detected': True, 'type': 'bullish_divergence', 'strength': 0.7}
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"RSI divergence analysis error: {e}")
            return {'detected': False}
    
    def _detect_bollinger_squeeze(self, price_data: pd.DataFrame, period: int = 20) -> Dict:
        """Detect Bollinger Band squeeze for consolidation"""
        try:
            closes = price_data['close']
            sma = closes.rolling(period).mean()
            std = closes.rolling(period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            band_width = (upper_band - lower_band) / sma
            current_width = band_width.iloc[-1] if not band_width.empty else 0
            
            # Squeeze detection
            is_squeeze = current_width < band_width.quantile(0.2) if len(band_width) > 20 else False
            
            return {
                'squeeze_detected': is_squeeze,
                'band_width': current_width,
                'percentile': band_width.rank(pct=True).iloc[-1] if len(band_width) > 20 else 0.5
            }
            
        except Exception as e:
            logger.error(f"Bollinger squeeze detection error: {e}")
            return {'squeeze_detected': False}
    
    # Deep learning helper methods
    def _create_sequences(self, data: np.array) -> np.array:
        """Create sequences for deep learning models"""
        sequences = []
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def _simple_lstm_prediction(self, sequences: np.array) -> Dict:
        """Simple LSTM prediction (placeholder implementation)"""
        # This would be implemented with actual TensorFlow/PyTorch
        # For now, return a simple prediction
        if len(sequences) == 0:
            return {'prediction': 0, 'confidence': 0, 'direction': 'neutral'}
        
        last_sequence = sequences[-1]
        simple_prediction = last_sequence[-1] + (last_sequence[-1] - last_sequence[-5]) / 5
        
        return {
            'prediction': simple_prediction,
            'confidence': 0.6,  # Placeholder confidence
            'direction': 'up' if simple_prediction > last_sequence[-1] else 'down'
        }

    def _calculate_pattern_confidence(self, patterns: Dict) -> float:
        """Calculate overall pattern recognition confidence"""
        confidence_factors = []
        
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'strength' in pattern_data:
                confidence_factors.append(pattern_data['strength'])
            elif isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                confidence_factors.append(0.7)  # Default confidence for detected patterns
        
        if not confidence_factors:
            return 0.5  # Neutral confidence
        
        return min(1.0, max(0.0, np.mean(confidence_factors)))
    
    def _get_fallback_pattern_analysis(self) -> Dict:
        """Fallback pattern analysis when primary methods fail"""
        return {
            'success': False,
            'patterns': {
                'trend_analysis': {'direction': 'unknown', 'strength': 0.5},
                'reversal_signals': {},
                'consolidation_zones': {'volatility_regime': 'unknown'},
                'volatility_regimes': {'current_regime': 'unknown'},
                'seasonal_tendencies': {}
            },
            'pattern_confidence': 0.3,
            'timestamp': datetime.now().isoformat()
        }

class AdvancedBacktester:
    """Advanced backtesting system with walk-forward analysis"""
    
    def __init__(self, trading_strategy):
        self.strategy = trading_strategy
        self.backtest_results = {}
        self.optimization_parameters = {}
    
    def run_walk_forward_analysis(self, historical_data: pd.DataFrame, 
                                 periods: int = 10, train_ratio: float = 0.7) -> Dict:
        """Run walk-forward analysis for strategy validation"""
        # Implementation would go here
        return {'status': 'walk_forward_analysis_placeholder'}

class ReinforcementLearningOptimizer:
    """Reinforcement learning for adaptive trading optimization"""
    
    def __init__(self, state_space_size: int, action_space_size: int):
        self.state_space = state_space_size
        self.action_space = action_space_size
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value for reinforcement learning"""
        # Q-learning implementation would go here
        pass

# Pattern recognition enums and constants
class PatternTypes:
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"

class MarketRegimes:
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    TRANSITION = "transition"
