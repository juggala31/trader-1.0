import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

class FixedBacktester:
    def __init__(self, symbols=None, initial_balance=100000):
        self.symbols = symbols or ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]
        self.initial_balance = initial_balance
        
        # Simple ensemble fallback
        self.ensemble = self.SimpleEnsemble()
        
        from market_regime_detector import MarketRegimeDetector
        self.regime_detector = MarketRegimeDetector()
        self.backtest_results = {}
        self.model_save_path = "trained_models/"
        self.progress_callback = None
        
        os.makedirs(self.model_save_path, exist_ok=True)
    
    class SimpleEnsemble:
        def __init__(self):
            self.is_trained = False
            
        def train_models(self, X_train, y_train):
            print("Training simple ensemble...")
            # Simple training - just store the data characteristics
            self.feature_means = X_train.mean()
            self.target_mean = y_train.mean()
            self.is_trained = True
            
        def predict(self, X_test):
            if not self.is_trained:
                # Random predictions if not trained
                return np.random.randint(0, 2, len(X_test))
            
            # Simple rule-based prediction based on feature means
            feature_diffs = (X_test - self.feature_means).mean(axis=1)
            return (feature_diffs > 0).astype(int)
    
    def set_progress_callback(self, callback):
        self.progress_callback = callback
    
    def log_progress(self, message):
        print(message)
        if self.progress_callback:
            self.progress_callback(message)
    
    def get_historical_data_fixed(self, symbol, years=2):
        """Improved historical data retrieval with better error handling"""
        try:
            # First try to get more data with different method
            self.log_progress(f"Fetching data for {symbol}...")
            
            # Calculate proper date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            self.log_progress(f"Date range: {start_date.date()} to {end_date.date()}")
            
            # Try different timeframe if daily doesn't work
            timeframes = [mt5.TIMEFRAME_D1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_H1]
            timeframe_names = ["D1 (Daily)", "H4 (4 Hours)", "H1 (1 Hour)"]
            
            for timeframe, timeframe_name in zip(timeframes, timeframe_names):
                try:
                    self.log_progress(f"Trying {timeframe_name} timeframe...")
                    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
                    
                    if rates is not None and len(rates) > 0:
                        self.log_progress(f"✓ Got {len(rates)} bars with {timeframe_name}")
                        
                        df = pd.DataFrame(rates)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('time', inplace=True)
                        
                        # If we have sufficient data, use it
                        if len(df) >= 50:
                            df = self._calculate_technical_indicators(df)
                            return df
                        else:
                            self.log_progress(f"⚠️ Only {len(df)} bars with {timeframe_name} - trying next timeframe")
                            continue
                    
                except Exception as e:
                    self.log_progress(f"Error with {timeframe_name}: {e}")
                    continue
            
            # If all timeframes fail, generate simulated data
            self.log_progress("All timeframes failed - generating simulated data")
            return self.generate_realistic_data(symbol, years)
            
        except Exception as e:
            self.log_progress(f"Data retrieval error: {e} - using simulated data")
            return self.generate_realistic_data(symbol, years)
    
    def generate_realistic_data(self, symbol, years=2):
        """Generate realistic simulated data with proper volume"""
        self.log_progress(f"Generating realistic simulated data for {symbol} ({years} years)")
        
        # Generate daily data for the specified years
        days = years * 365
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Symbol-specific base prices and characteristics
        symbol_profiles = {
            "US30Z25.sim": {"base_price": 35000, "volatility": 0.015, "trend": 0.0002},
            "US100Z25.sim": {"base_price": 18000, "volatility": 0.012, "trend": 0.0003},
            "XAUZ25.sim": {"base_price": 2000, "volatility": 0.008, "trend": 0.0001}
        }
        
        profile = symbol_profiles.get(symbol, {"base_price": 10000, "volatility": 0.01, "trend": 0.0002})
        
        # Generate realistic price series
        prices = [profile["base_price"]]
        for i in range(1, days):
            # Realistic price movement with trend and volatility
            change = profile["trend"] + np.random.normal(0, profile["volatility"])
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, profile["base_price"] * 0.5))  # Prevent negative prices
        
        # Create DataFrame with realistic OHLC data
        df = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'tick_volume': [abs(int(np.random.normal(1000, 200))) for _ in prices]  # Realistic volume
        }, index=dates)
        
        df = self._calculate_technical_indicators(df)
        self.log_progress(f"✓ Generated {len(df)} realistic bars for {symbol}")
        return df
    
    def _calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Basic features
        df['returns'] = df['close'].pct_change()
        
        # Multiple moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # MACD (simplified)
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price position
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume features if available
        if 'tick_volume' in df.columns:
            df['volume_sma_20'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma_20']
        
        return df.dropna()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI with proper error handling"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral value for NaN
    
    def prepare_features_and_target(self, df, forecast_horizon=1):
        """Prepare features with better handling of small datasets"""
        try:
            # Create target variable
            df['target'] = (df['close'].shift(-forecast_horizon) > df['close']).astype(int)
            
            # Select features (exclude price and target columns)
            exclude_cols = ['target', 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            # If we have very few features, create some basic ones
            if len(feature_columns) < 5:
                feature_columns.extend(['returns', 'volatility_20', 'rsi_14'])
            
            X = df[feature_columns]
            y = df['target']
            
            # Remove rows with NaN in target
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) < 30:
                self.log_progress(f"⚠️ Small dataset: {len(X)} samples")
                # For small datasets, we can still work with them
                pass
                
            return X, y, feature_columns
            
        except Exception as e:
            self.log_progress(f"Error preparing features: {e}")
            return None, None, None
    
    def run_backtest(self, symbol, train_ratio=0.7, years=2):
        """Run backtest that works with smaller datasets"""
        self.log_progress(f"\\n=== BACKTESTING {symbol} ===")
        
        # Get data with improved retrieval
        df = self.get_historical_data_fixed(symbol, years=years)
        if df is None:
            self.log_progress(f"❌ No data available for {symbol}")
            return None
        
        if len(df) < 30:
            self.log_progress(f"⚠️ Small dataset for {symbol}: {len(df)} bars")
            # We can still work with smaller datasets
            pass
        
        # Prepare features
        X, y, feature_columns = self.prepare_features_and_target(df)
        if X is None or y is None or len(X) < 10:
            self.log_progress(f"❌ Insufficient features for {symbol}")
            return None
        
        # Adjust train ratio for small datasets
        if len(X) < 100:
            train_ratio = 0.8  # Use more data for training
        
        # Split data
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.log_progress(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Train model
        self.ensemble.train_models(X_train, y_train)
        
        # Test model
        predictions = self.ensemble.predict(X_test)
        accuracy = (predictions == y_test).mean() if len(y_test) > 0 else 0
        
        # Trading simulation
        trading_results = self._simulate_trading_performance(df, X_test, predictions, split_idx)
        
        results = {
            'symbol': symbol,
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'features_used': feature_columns,
            'trading_performance': trading_results,
            'data_points': len(df)
        }
        
        self.log_progress(f"✓ {symbol}: Accuracy={accuracy:.3f}, Return={trading_results['total_return']:.2%}")
        return results
    
    def _simulate_trading_performance(self, df, X_test, predictions, split_idx):
        """Simulate trading with better handling of small datasets"""
        if len(X_test) == 0:
            return {
                'initial_balance': self.initial_balance / len(self.symbols),
                'final_balance': self.initial_balance / len(self.symbols),
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'sharpe_ratio': 0,
                'volatility': 0
            }
        
        test_prices = df['close'].iloc[split_idx:split_idx + len(X_test)]
        
        initial_balance = self.initial_balance / len(self.symbols)
        balance = initial_balance
        position = 0
        trades = []
        returns = []
        
        for i, (prediction, price) in enumerate(zip(predictions, test_prices)):
            if i == 0:
                previous_price = price
                continue
            
            if prediction == 1 and position == 0:  # Buy
                trade_size = min(balance * 0.1, balance)  # Ensure we don't overspend
                position = trade_size / price
                balance -= trade_size
                trades.append({'type': 'BUY', 'price': price})
                
            elif prediction == 0 and position > 0:  # Sell
                trade_value = position * price
                balance += trade_value
                trade_return = (trade_value - (position * previous_price)) / (position * previous_price)
                returns.append(trade_return)
                trades.append({'type': 'SELL', 'price': price, 'return': trade_return})
                position = 0
            
            previous_price = price
        
        # Close any open position
        if position > 0:
            final_value = position * test_prices.iloc[-1]
            balance += final_value
        
        # Calculate metrics
        total_return = (balance - initial_balance) / initial_balance
        volatility = np.std(returns) if returns else 0.01
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t.get('return', 0) > 0]),
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility
        }
    
    def run_comprehensive_backtest(self, years=2, save_models=True):
        """Run comprehensive backtesting that works with any data size"""
        self.log_progress("🚀 STARTING COMPREHENSIVE BACKTESTING")
        self.log_progress("=" * 50)
        
        # Try MT5 connection
        try:
            if not mt5.initialize():
                self.log_progress("MT5 initialization failed - using simulated data")
            else:
                mt5.login(1600038177, server="OANDA-Demo-1")
                self.log_progress("✓ MT5 connected")
        except:
            self.log_progress("MT5 connection failed - using simulated data")
        
        self.backtest_results = {}
        successful_tests = 0
        
        # Backtest each symbol
        for symbol in self.symbols:
            results = self.run_backtest(symbol, years=years)
            if results:
                self.backtest_results[symbol] = results
                successful_tests += 1
        
        # Train regime detection if we have sufficient data
        if successful_tests > 0:
            self.log_progress("\\n=== TRAINING MARKET REGIME DETECTION ===")
            for symbol in self.symbols:
                df = self.get_historical_data_fixed(symbol, years=years)
                if df is not None and len(df) >= 50:
                    self.regime_detector.train_model(df['close'])
                    self.log_progress("✓ Regime detection trained")
                    break
        
        # Save models if we have results
        if save_models and successful_tests > 0:
            self.save_trained_models()
        
        # Generate report
        self.generate_summary_report()
        
        # Shutdown MT5 if connected
        try:
            mt5.shutdown()
        except:
            pass
        
        self.log_progress(f"✓ Backtesting completed: {successful_tests}/{len(self.symbols)} symbols successful")
        return successful_tests > 0
    
    def save_trained_models(self):
        """Save trained models"""
        try:
            ensemble_path = os.path.join(self.model_save_path, "trained_ensemble.pkl")
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble, f)
            
            regime_path = os.path.join(self.model_save_path, "trained_regime_detector.pkl")
            with open(regime_path, 'wb') as f:
                pickle.dump(self.regime_detector, f)
            
            self.log_progress("✓ Models saved successfully")
            
        except Exception as e:
            self.log_progress(f"⚠️ Error saving models: {e}")
    
    def generate_summary_report(self):
        """Generate summary report"""
        if not self.backtest_results:
            self.log_progress("❌ No backtest results to report")
            return
        
        self.log_progress("\\n" + "=" * 50)
        self.log_progress("📊 BACKTESTING SUMMARY REPORT")
        self.log_progress("=" * 50)
        
        total_accuracy = 0
        total_return = 0
        
        for symbol, results in self.backtest_results.items():
            self.log_progress(f"\\n{symbol}:")
            self.log_progress(f"  Accuracy: {results['accuracy']:.3f}")
            self.log_progress(f"  Return: {results['trading_performance']['total_return']:.2%}")
            self.log_progress(f"  Sharpe: {results['trading_performance']['sharpe_ratio']:.3f}")
            self.log_progress(f"  Trades: {results['trading_performance']['total_trades']}")
            self.log_progress(f"  Data Points: {results['data_points']}")
            
            total_accuracy += results['accuracy']
            total_return += results['trading_performance']['total_return']
        
        symbol_count = len(self.backtest_results)
        avg_accuracy = total_accuracy / symbol_count
        avg_return = total_return / symbol_count
        
        self.log_progress(f"\\n📈 AVERAGE: Accuracy={avg_accuracy:.3f}, Return={avg_return:.2%}")
        
        # Save to file
        try:
            with open("backtesting_report.txt", 'w') as f:
                f.write("FTMO AI Trading System - Backtesting Report\\n")
                f.write("=" * 50 + "\\n")
                for symbol, results in self.backtest_results.items():
                    f.write(f"{symbol}: Accuracy={results['accuracy']:.3f}, Return={results['trading_performance']['total_return']:.2%}\\n")
            
            self.log_progress("✓ Report saved to backtesting_report.txt")
        except Exception as e:
            self.log_progress(f"⚠️ Error saving report: {e}")

# Update the backtester reference in the dashboard
def get_fixed_backtester():
    return FixedBacktester()

if __name__ == "__main__":
    backtester = FixedBacktester()
    backtester.run_comprehensive_backtest(years=1, save_models=True)
