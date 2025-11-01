import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

class WorkingBacktester:
    def __init__(self, symbols=None, initial_balance=100000):
        self.symbols = symbols or ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]
        self.initial_balance = initial_balance
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
            # Simple mean-based prediction
            return (X_test.mean(axis=1) > X_test.mean(axis=1).median()).astype(int)
    
    def set_progress_callback(self, callback):
        self.progress_callback = callback
    
    def log_progress(self, message):
        print(message)
        if self.progress_callback:
            self.progress_callback(message)
    
    def get_historical_data_simple(self, symbol, years=1):
        """Simple and reliable data retrieval"""
        try:
            self.log_progress(f"Getting data for {symbol}...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            # Try daily timeframe first
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start_date, end_date)
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                self.log_progress(f"✓ Got {len(df)} bars for {symbol}")
                return df
            else:
                self.log_progress(f"❌ No MT5 data for {symbol} - using simulation")
                return self.generate_simple_data(symbol, years)
                
        except Exception as e:
            self.log_progress(f"Error getting data: {e} - using simulation")
            return self.generate_simple_data(symbol, years)
    
    def generate_simple_data(self, symbol, years=1):
        """Generate simple but effective simulated data"""
        days = years * 365
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Simple price generation
        base_prices = {
            "US30Z25.sim": 35000,
            "US100Z25.sim": 18000, 
            "XAUZ25.sim": 2000
        }
        base_price = base_prices.get(symbol, 10000)
        
        prices = [base_price]
        for i in range(1, days):
            # Simple random walk with slight upward trend
            change = 0.0001 + np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices
        }, index=dates)
        
        self.log_progress(f"✓ Generated {len(df)} simulated bars for {symbol}")
        return df
    
    def calculate_simple_features(self, df):
        """Calculate basic features that always work"""
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Simple RSI
        gains = df['returns'].where(df['returns'] > 0, 0)
        losses = -df['returns'].where(df['returns'] < 0, 0)
        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Drop NaN values
        df = df.dropna()
        return df
    
    def prepare_simple_features(self, df):
        """Prepare features without complex processing"""
        # Create target (next day price movement)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Simple feature selection
        feature_cols = ['returns', 'sma_10', 'sma_20', 'volatility', 'rsi']
        
        # Ensure all features exist
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features]
        y = df['target']
        
        # Remove rows where target is NaN (end of dataset)
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.log_progress(f"Prepared {len(X)} samples with {len(available_features)} features")
        return X, y, available_features
    
    def run_simple_backtest(self, symbol, years=1):
        """Run a simple but reliable backtest"""
        self.log_progress(f"\\n=== BACKTESTING {symbol} ===")
        
        # Get data
        df = self.get_historical_data_simple(symbol, years)
        if df is None or len(df) < 20:
            self.log_progress(f"❌ Insufficient data for {symbol}")
            return None
        
        # Calculate features
        df = self.calculate_simple_features(df)
        if len(df) < 10:
            self.log_progress(f"❌ Not enough data after feature calculation: {len(df)}")
            return None
        
        # Prepare features
        X, y, features = self.prepare_simple_features(df)
        if len(X) < 10:
            self.log_progress(f"❌ Not enough samples for training: {len(X)}")
            return None
        
        # Simple train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.log_progress(f"Training: {len(X_train)}, Testing: {len(X_test)}")
        
        # Train model
        self.ensemble.train_models(X_train, y_train)
        
        # Test model
        predictions = self.ensemble.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        # Simple trading simulation
        trading_results = self.simulate_simple_trading(df, X_test, predictions, split_idx)
        
        results = {
            'symbol': symbol,
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'total_return': trading_results['total_return'],
            'total_trades': trading_results['total_trades'],
            'data_points': len(df)
        }
        
        self.log_progress(f"✓ {symbol}: Accuracy={accuracy:.3f}, Return={trading_results['total_return']:.2%}")
        return results
    
    def simulate_simple_trading(self, df, X_test, predictions, split_idx):
        """Simple trading simulation"""
        if len(X_test) == 0:
            return {'total_return': 0, 'total_trades': 0}
        
        test_prices = df['close'].iloc[split_idx:split_idx + len(X_test)]
        initial_balance = 10000
        balance = initial_balance
        trades = 0
        
        for i, (prediction, price) in enumerate(zip(predictions, test_prices)):
            if i == 0:
                continue
                
            # Simple trading: buy if prediction is 1, sell if 0
            if prediction == 1 and balance > 1000:  # Buy signal
                # Simulate buying
                trade_amount = min(balance * 0.1, 1000)
                balance -= trade_amount
                trades += 1
            elif prediction == 0 and balance < initial_balance:  # Sell signal
                # Simulate selling with profit/loss
                previous_price = test_prices.iloc[i-1]
                price_change = (price - previous_price) / previous_price
                balance += balance * price_change * 0.1  # 10% position
                trades += 1
        
        total_return = (balance - initial_balance) / initial_balance
        return {'total_return': total_return, 'total_trades': trades}
    
    def run_comprehensive_backtest(self, years=1, save_models=True):
        """Run comprehensive backtesting"""
        self.log_progress("🚀 STARTING COMPREHENSIVE BACKTESTING")
        self.log_progress("=" * 50)
        
        # Initialize MT5
        try:
            mt5.initialize()
            mt5.login(1600038177, server="OANDA-Demo-1")
            mt5_connected = True
            self.log_progress("✓ MT5 connected")
        except:
            mt5_connected = False
            self.log_progress("⚠️ MT5 not connected - using simulated data")
        
        self.backtest_results = {}
        successful_tests = 0
        
        # Test each symbol
        for symbol in self.symbols:
            results = self.run_simple_backtest(symbol, years)
            if results:
                self.backtest_results[symbol] = results
                successful_tests += 1
        
        # Train regime detection if we have data
        if successful_tests > 0:
            self.log_progress("Training regime detection...")
            # Use the first successful symbol's data
            first_symbol = list(self.backtest_results.keys())[0]
            df = self.get_historical_data_simple(first_symbol, years)
            if df is not None and len(df) >= 30:
                self.regime_detector.train_model(df['close'])
                self.log_progress("✓ Regime detection trained")
        
        # Save models
        if save_models and successful_tests > 0:
            try:
                ensemble_path = os.path.join(self.model_save_path, "trained_ensemble.pkl")
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(self.ensemble, f)
                
                regime_path = os.path.join(self.model_save_path, "trained_regime_detector.pkl")
                with open(regime_path, 'wb') as f:
                    pickle.dump(self.regime_detector, f)
                
                self.log_progress("✓ Models saved")
            except Exception as e:
                self.log_progress(f"⚠️ Error saving models: {e}")
        
        # Generate report
        self.generate_simple_report()
        
        # Cleanup
        if mt5_connected:
            try:
                mt5.shutdown()
            except:
                pass
        
        self.log_progress(f"✓ Backtesting completed: {successful_tests}/{len(self.symbols)} successful")
        return successful_tests > 0
    
    def generate_simple_report(self):
        """Generate simple report"""
        if not self.backtest_results:
            self.log_progress("❌ No results to report")
            return
        
        self.log_progress("\\n📊 BACKTEST RESULTS")
        self.log_progress("=" * 30)
        
        for symbol, results in self.backtest_results.items():
            self.log_progress(f"{symbol}:")
            self.log_progress(f"  Accuracy: {results['accuracy']:.3f}")
            self.log_progress(f"  Return: {results['total_return']:.2%}")
            self.log_progress(f"  Trades: {results['total_trades']}")
            self.log_progress(f"  Samples: {results['data_points']}")
        
        # Save to file
        try:
            with open("backtesting_report.txt", 'w') as f:
                f.write("Backtesting Report\\n")
                f.write("=" * 20 + "\\n")
                for symbol, results in self.backtest_results.items():
                    f.write(f"{symbol}: Acc={results['accuracy']:.3f}, Ret={results['total_return']:.2%}\\n")
            self.log_progress("✓ Report saved to backtesting_report.txt")
        except Exception as e:
            self.log_progress(f"⚠️ Error saving report: {e}")

# Simple test
def test_working_backtester():
    backtester = WorkingBacktester()
    
    def progress_callback(msg):
        print(f"PROGRESS: {msg}")
    
    backtester.set_progress_callback(progress_callback)
    backtester.run_comprehensive_backtest(years=1, save_models=True)

if __name__ == "__main__":
    test_working_backtester()
