import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

class ImprovedBacktester:
    def __init__(self, symbols=None, initial_balance=100000):
        self.symbols = symbols or ['BTCX25.sim', 'US30Z25.sim', 'XAUZ25.sim', 'US100Z25.sim', 'US500Z25.sim', 'USOILZ25.sim']
        self.initial_balance = initial_balance
        
        # Try to import your ensemble, or create a simple one
        try:
            from models.ensemble.professional_ensemble import ProfessionalEnsemble
            self.ensemble = ProfessionalEnsemble()
            print("✓ ProfessionalEnsemble imported")
        except ImportError:
            print("⚠️  Using simple ensemble fallback")
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
            self.is_trained = True
            
        def predict(self, X_test):
            if not self.is_trained:
                return np.random.randint(0, 2, len(X_test))
            # Simple rule-based prediction for demo
            return (X_test.mean(axis=1) > X_test.mean(axis=1).median()).astype(int)
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def log_progress(self, message):
        """Log progress message"""
        print(message)
        if self.progress_callback:
            self.progress_callback(message)
    
    def connect_to_mt5(self):
        """Connect to MT5 with better error handling"""
        try:
            if not mt5.initialize():
                self.log_progress("MT5 initialization failed - using simulated data")
                return False
            
            authorized = mt5.login(1600038177, server="OANDA-Demo-1")
            if not authorized:
                self.log_progress("MT5 login failed - using simulated data")
                return False
            
            self.log_progress("✓ Connected to MT5")
            return True
            
        except Exception as e:
            self.log_progress(f"MT5 connection error: {e} - using simulated data")
            return False
    
    def generate_simulated_data(self, symbol, years=2):
        """Generate realistic simulated data when MT5 is unavailable"""
        self.log_progress(f"Generating simulated data for {symbol}")
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=years*365), 
                             end=datetime.now(), freq='D')
        
        # Create more realistic price series with trends and volatility
        prices = [100]
        for i in range(1, len(dates)):
            # Realistic price movement with some trends
            trend = 0.0001  # Small upward trend
            volatility = 0.01  # Realistic volatility
            change = trend + np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]
        }, index=dates)
        
        df = self._calculate_technical_indicators(df)
        self.log_progress(f"✓ Generated {len(df)} simulated bars for {symbol}")
        return df
    
    def get_historical_data(self, symbol, timeframe=mt5.TIMEFRAME_D1, years=2):
        """Get historical data with fallback to simulated data"""
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            if rates is None or len(rates) == 0:
                self.log_progress(f"No MT5 data for {symbol} - using simulated data")
                return self.generate_simulated_data(symbol, years)
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            df = self._calculate_technical_indicators(df)
            self.log_progress(f"✓ Retrieved {len(df)} real bars for {symbol}")
            return df
            
        except Exception as e:
            self.log_progress(f"Error getting data for {symbol}: {e} - using simulated data")
            return self.generate_simulated_data(symbol, years)
    
    def _calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        # Basic features
        df['returns'] = df['close'].pct_change()
        
        # Moving averages
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # Price position
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df.dropna()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral 50
    
    def prepare_features_and_target(self, df, forecast_horizon=1):
        """Prepare features and target with robust error handling"""
        try:
            # Create target variable
            df['target'] = (df['close'].shift(-forecast_horizon) > df['close']).astype(int)
            
            # Select features (exclude price columns)
            exclude_cols = ['target', 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_columns]
            y = df['target']
            
            # Remove rows with NaN
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) == 0:
                self.log_progress("Warning: No valid data after processing")
                return None, None, None
                
            return X, y, feature_columns
            
        except Exception as e:
            self.log_progress(f"Error preparing features: {e}")
            return None, None, None
    
    def run_backtest(self, symbol, train_ratio=0.7, years=2):
        """Run backtest with comprehensive error handling"""
        self.log_progress(f"\\n=== BACKTESTING {symbol} ===")
        
        # Get data
        df = self.get_historical_data(symbol, years=years)
        if df is None or len(df) < 50:
            self.log_progress(f"⚠️ Insufficient data for {symbol}")
            return None
        
        # Prepare features
        X, y, feature_columns = self.prepare_features_and_target(df)
        if X is None or y is None:
            self.log_progress(f"⚠️ Could not prepare features for {symbol}")
            return None
        
        if len(X) < 30:
            self.log_progress(f"⚠️ Not enough samples for {symbol}: {len(X)}")
            return None
        
        # Split data
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.log_progress(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")
        self.log_progress(f"Features: {len(feature_columns)}")
        
        # Train model
        self.log_progress("Training model...")
        self.ensemble.train_models(X_train, y_train)
        
        # Test model
        self.log_progress("Testing model...")
        predictions = self.ensemble.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        # Trading simulation
        trading_results = self._simulate_trading_performance(df, X_test, predictions, split_idx)
        
        results = {
            'symbol': symbol,
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'features_used': feature_columns,
            'trading_performance': trading_results
        }
        
        self.log_progress(f"✓ {symbol}: Accuracy={accuracy:.3f}, Return={trading_results['total_return']:.2%}")
        return results
    
    def _simulate_trading_performance(self, df, X_test, predictions, split_idx):
        """Simulate trading performance"""
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
                trade_size = balance * 0.1
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
    
    def train_regime_detection(self, symbols=None, years=2):
        """Train market regime detection"""
        self.log_progress("\\n=== TRAINING MARKET REGIME DETECTION ===")
        
        symbols = symbols or self.symbols
        
        for symbol in symbols:
            df = self.get_historical_data(symbol, years=years)
            if df is not None:
                price_data = df['close']
                self.regime_detector.train_model(price_data)
                self.log_progress("✓ Regime detection trained")
                return True
        
        self.log_progress("⚠️ Could not train regime detection")
        return False
    
    def run_comprehensive_backtest(self, years=2, save_models=True):
        """Run comprehensive backtesting"""
        self.log_progress("🚀 STARTING COMPREHENSIVE BACKTESTING")
        self.log_progress("=" * 50)
        
        # Try MT5 connection
        mt5_connected = self.connect_to_mt5()
        
        self.backtest_results = {}
        successful_tests = 0
        
        # Backtest each symbol
        for symbol in self.symbols:
            results = self.run_backtest(symbol, years=years)
            if results:
                self.backtest_results[symbol] = results
                successful_tests += 1
        
        # Train regime detection
        if successful_tests > 0:
            self.train_regime_detection(years=years)
        
        # Save models if we have results
        if save_models and successful_tests > 0:
            self.save_trained_models()
        
        # Generate report
        self.generate_summary_report()
        
        if mt5_connected:
            mt5.shutdown()
        
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
            self.log_progress("⚠️ No backtest results to report")
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

def run_improved_backtesting():
    """Main function for improved backtesting"""
    print("🎯 IMPROVED BACKTESTING SYSTEM")
    print("=" * 60)
    
    backtester = ImprovedBacktester()
    
    # Set progress callback
    def progress_callback(message):
        print(message)
    
    backtester.set_progress_callback(progress_callback)
    
    # Run backtest
    success = backtester.run_comprehensive_backtest(years=1, save_models=True)
    
    if success:
        print("\\n🎉 BACKTESTING COMPLETED SUCCESSFULLY!")
    else:
        print("\\n⚠️  Backtesting completed with limited results")

if __name__ == "__main__":
    run_improved_backtesting()

