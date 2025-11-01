# advanced_features.py - Professional Feature Engineering
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger('FTMO_AI')

class AdvancedFeatureEngineer:
    """Professional feature engineering for trading AI"""
    
    def __init__(self):
        self.feature_cache = {}
        
    def extract_professional_features(self, symbol: str, df: pd.DataFrame, current_price: float) -> dict:
        """Extract 40+ professional trading features"""
        if len(df) < 50:
            return self._get_fallback_features()
            
        features = {}
        
        try:
            # 1. Enhanced Technical Indicators
            features.update(self._get_enhanced_technical_features(df, current_price))
            
            # 2. Market Microstructure
            features.update(self._get_microstructure_features(df))
            
            # 3. Volatility & Regime Analysis
            features.update(self._get_volatility_features(df))
            
            # 4. Price Action Patterns
            features.update(self._get_price_action_features(df, current_price))
            
            # 5. Temporal & Seasonal Patterns
            features.update(self._get_temporal_features())
            
            # 6. Symbol-Specific Features
            features.update(self._get_symbol_specific_features(symbol))
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return self._get_fallback_features()
        
        return features
    
    def _get_enhanced_technical_features(self, df: pd.DataFrame, current_price: float) -> dict:
        """Enhanced technical indicators with multiple timeframes"""
        features = {}
        
        # Multi-period RSI (5, 14, 21, 50)
        for period in [5, 14, 21, 50]:
            features[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # EMA convergence/divergence with multiple pairs
        ema_pairs = [(5, 20), (8, 21), (13, 34), (21, 55), (50, 200)]
        for fast, slow in ema_pairs:
            if len(df) >= slow:
                ema_fast = df['close'].ewm(span=fast).mean().iloc[-1]
                ema_slow = df['close'].ewm(span=slow).mean().iloc[-1]
                features[f'ema_ratio_{fast}_{slow}'] = ema_fast / ema_slow
                features[f'ema_diff_{fast}_{slow}'] = (ema_fast - ema_slow) / ema_slow * 100
        
        # Advanced MACD with multiple components
        macd_short, macd_long, macd_signal = 12, 26, 9
        if len(df) >= macd_long:
            exp1 = df['close'].ewm(span=macd_short).mean()
            exp2 = df['close'].ewm(span=macd_long).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=macd_signal).mean()
            histogram = macd - signal
            
            features.update({
                'macd_value': float(macd.iloc[-1]),
                'macd_signal': float(signal.iloc[-1]),
                'macd_histogram': float(histogram.iloc[-1]),
                'macd_trend': float(macd.iloc[-1] - macd.iloc[-10]) if len(macd) >= 10 else 0.0
            })
        
        # Bollinger Bands
        if len(df) >= 20:
            bb_period = 20
            sma = df['close'].rolling(bb_period).mean()
            std = df['close'].rolling(bb_period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            features.update({
                'bb_position': (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]),
                'bb_width': (upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1] * 100,
                'bb_squeeze': 1 if features.get('bb_width', 0) < 2.0 else 0
            })
        
        return features
    
    def _get_microstructure_features(self, df: pd.DataFrame) -> dict:
        """Market microstructure analysis"""
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return self._get_microstructure_fallback()
        
        features = {
            # Volatility measures
            'volatility_5m': returns.iloc[-5:].std() * 100,
            'volatility_15m': returns.iloc[-15:].std() * 100,
            'volatility_1h': returns.iloc[-60:].std() * 100 if len(returns) >= 60 else returns.std() * 100,
            'volatility_4h': returns.iloc[-240:].std() * 100 if len(returns) >= 240 else returns.std() * 100,
            'volatility_ratio_short_long': returns.iloc[-15:].std() / returns.iloc[-60:].std() if len(returns) >= 60 else 1.0,
            
            # Momentum and acceleration
            'momentum_5': returns.iloc[-5:].sum() * 100,
            'momentum_15': returns.iloc[-15:].sum() * 100,
            'acceleration': returns.iloc[-1] - returns.iloc[-5] if len(returns) >= 5 else 0.0,
            
            # Statistical properties
            'returns_skewness': returns.skew() if len(returns) > 10 else 0.0,
            'returns_kurtosis': returns.kurtosis() if len(returns) > 10 else 0.0,
            'returns_autocorr': returns.autocorr() if len(returns) > 10 else 0.0,
            
            # Range-based features
            'true_range': self._calculate_average_true_range(df, 14) if len(df) >= 14 else 0.0,
            'adx': self._calculate_adx(df, 14) if len(df) >= 28 else 0.0,
        }
        
        return features
    
    def _get_volatility_features(self, df: pd.DataFrame) -> dict:
        """Volatility regime and clustering features"""
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 20:
            return {'volatility_regime': 2, 'volatility_cluster': 0.5, 'regime_stability': 0.5}
        
        # GARCH-like volatility estimation (simplified)
        recent_vol = returns.iloc[-10:].std()
        medium_vol = returns.iloc[-20:].std()
        long_vol = returns.iloc[-50:].std() if len(returns) >= 50 else medium_vol
        
        # Volatility regime (1=low, 2=medium, 3=high)
        if recent_vol < long_vol * 0.7:
            vol_regime = 1
        elif recent_vol > long_vol * 1.3:
            vol_regime = 3
        else:
            vol_regime = 2
        
        # Volatility clustering (recent vs historical)
        vol_cluster = recent_vol / long_vol if long_vol > 0 else 1.0
        
        # Regime stability (how consistent is the current regime)
        regime_stability = 1.0 - abs(recent_vol - medium_vol) / medium_vol if medium_vol > 0 else 0.5
        
        return {
            'volatility_regime': vol_regime,
            'volatility_cluster': vol_cluster,
            'regime_stability': max(0.0, min(1.0, regime_stability)),
            'volatility_momentum': (recent_vol - medium_vol) / medium_vol if medium_vol > 0 else 0.0
        }
    
    def _get_price_action_features(self, df: pd.DataFrame, current_price: float) -> dict:
        """Price action and pattern recognition"""
        features = {}
        
        if len(df) < 20:
            return features
        
        # Support and resistance levels
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        mid_point = (recent_high + recent_low) / 2
        
        features.update({
            'price_vs_high': (current_price - recent_high) / recent_high * 100,
            'price_vs_low': (current_price - recent_low) / recent_low * 100,
            'price_vs_mid': (current_price - mid_point) / mid_point * 100,
            'high_low_range': (recent_high - recent_low) / recent_low * 100,
        })
        
        # Price position in recent range
        if recent_high != recent_low:
            features['price_position'] = (current_price - recent_low) / (recent_high - recent_low)
        else:
            features['price_position'] = 0.5
        
        # Recent price patterns
        features['higher_highs'] = 1 if current_price > df['high'].iloc[-10:].max() else 0
        features['lower_lows'] = 1 if current_price < df['low'].iloc[-10:].min() else 0
        
        return features
    
    def _get_temporal_features(self) -> dict:
        """Time-based and seasonal features"""
        now = datetime.now()
        
        # Cyclical encoding for time features
        features = {
            # Hour of day (cyclical)
            'hour_sin': np.sin(2 * np.pi * now.hour / 24),
            'hour_cos': np.cos(2 * np.pi * now.hour / 24),
            
            # Day of week (cyclical)
            'day_sin': np.sin(2 * np.pi * now.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * now.weekday() / 7),
            
            # Month of year (cyclical)
            'month_sin': np.sin(2 * np.pi * now.month / 12),
            'month_cos': np.cos(2 * np.pi * now.month / 12),
            
            # Trading session flags
            'is_asia_session': 1.0 if 22 <= now.hour or now.hour <= 6 else 0.0,  # 10PM-6AM UTC
            'is_london_session': 1.0 if 7 <= now.hour <= 15 else 0.0,  # 7AM-3PM UTC
            'is_us_session': 1.0 if 13 <= now.hour <= 21 else 0.0,  # 1PM-9PM UTC
            'is_weekend': 1.0 if now.weekday() >= 5 else 0.0,
            
            # Month-specific (for seasonal patterns)
            'is_q1': 1.0 if now.month <= 3 else 0.0,
            'is_q2': 1.0 if 4 <= now.month <= 6 else 0.0,
            'is_q3': 1.0 if 7 <= now.month <= 9 else 0.0,
            'is_q4': 1.0 if now.month >= 10 else 0.0,
        }
        
        return features
    
    def _get_symbol_specific_features(self, symbol: str) -> dict:
        """Symbol-specific characteristics"""
        symbol_upper = symbol.upper()
        
        features = {
            'is_index': 1.0 if any(x in symbol_upper for x in ['US30', 'US100', 'SPX', 'DAX', 'FTSE']) else 0.0,
            'is_forex': 1.0 if any(x in symbol_upper for x in ['EUR', 'USD', 'GBP', 'JPY', 'AUD']) else 0.0,
            'is_commodity': 1.0 if any(x in symbol_upper for x in ['XAU', 'OIL', 'GOLD', 'SILVER']) else 0.0,
            'is_crypto': 1.0 if any(x in symbol_upper for x in ['BTC', 'ETH', 'XRP', 'ADA']) else 0.0,
            
            # Your specific symbols
            'is_us30': 1.0 if 'US30' in symbol_upper else 0.0,
            'is_us100': 1.0 if 'US100' in symbol_upper else 0.0,
            'is_xau': 1.0 if 'XAU' in symbol_upper else 0.0,
        }
        
        return features
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    
    def _calculate_average_true_range(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range"""
        if len(df) < period + 1:
            return 0.0
            
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not atr.empty else 0.0
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Average Directional Index (simplified)"""
        if len(df) < period * 2:
            return 0.0
            
        # Simplified ADX calculation
        high = df['high']
        low = df['low']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        # Basic directional movement
        dx = (plus_dm.rolling(period).mean() - minus_dm.rolling(period).mean()) / (
            plus_dm.rolling(period).mean() + minus_dm.rolling(period).mean()) * 100
        
        adx = dx.rolling(period).mean()
        return float(adx.iloc[-1]) if not adx.empty else 0.0
    
    def _get_fallback_features(self) -> dict:
        """Fallback features when data is insufficient"""
        return {
            'rsi_5': 50.0, 'rsi_14': 50.0, 'rsi_21': 50.0, 'rsi_50': 50.0,
            'volatility_regime': 2, 'volatility_cluster': 1.0, 'regime_stability': 0.5,
            'price_position': 0.5, 'higher_highs': 0, 'lower_lows': 0,
            'hour_sin': 0.0, 'hour_cos': 1.0, 'is_us_session': 0.0,
            'is_us30': 0.0, 'is_us100': 0.0, 'is_xau': 0.0
        }
    
    def _get_microstructure_fallback(self) -> dict:
        """Fallback microstructure features"""
        return {
            'volatility_5m': 1.0, 'volatility_15m': 1.0, 'volatility_1h': 1.0,
            'momentum_5': 0.0, 'momentum_15': 0.0, 'acceleration': 0.0,
            'returns_skewness': 0.0, 'returns_kurtosis': 0.0, 'true_range': 1.0
        }
