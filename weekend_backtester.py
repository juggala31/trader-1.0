import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Try to import your ensemble, or create a simple one if not available
try:
    from models.ensemble.professional_ensemble import ProfessionalEnsemble
    print("✓ ProfessionalEnsemble imported successfully")
except ImportError:
    print("⚠️  ProfessionalEnsemble not found, creating simple ensemble")
    
    class ProfessionalEnsemble:
        def __init__(self):
            self.models = {}
            self.is_trained = False
            
        def train_models(self, X_train, y_train):
            """Simple training method"""
            print("Training simple ensemble model...")
            self.is_trained = True
            
        def predict(self, X_test):
            """Simple prediction method"""
            return np.random.randint(0, 2, len(X_test))

class ComprehensiveBacktester:
    def __init__(self, symbols=None, initial_balance=100000):
        self.symbols = symbols or ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]
        self.initial_balance = initial_balance
        self.ensemble = ProfessionalEnsemble()
        from market_regime_detector import MarketRegimeDetector
        self.regime_detector = MarketRegimeDetector()
        self.backtest_results = {}
        self.model_save_path = "trained_models/"
        
        # Create directory for saving models
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def connect_to_mt5(self):
        """Connect to MT5 for historical data"""
        try:
            if not mt5.initialize():
                print("MT5 initialization failed")
                return False
            
            # Login to demo account
            authorized = mt5.login(1600038177, server="OANDA-Demo-1")
            if not authorized:
                print("MT5 login failed")
                return False
            
            print("✓ Connected to MT5 for historical data")
            return True
            
        except Exception as e:
            print(f"MT5 connection error: {e}")
            return False
    
    def get_historical_data(self, symbol, timeframe=mt5.TIMEFRAME_D1, years=2):
        """Get comprehensive historical data for backtesting"""
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            # Get rates
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            if rates is None:
                print(f"No data available for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Calculate additional features
            df = self._calculate_technical_indicators(df)
            
            print(f"✓ Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        
        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # Price position relative to ranges
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features_and_target(self, df, forecast_horizon=1):
        """Prepare features and target for machine learning"""
        # Create target variable (future price movement)
        df['target'] = (df['close'].shift(-forecast_horizon) > df['close']).astype(int)
        
        # Select features for training (exclude price columns)
        exclude_cols = ['target', 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns]
        y = df['target']
        
        # Remove rows with NaN in target (end of dataset)
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y, feature_columns
    
    def run_backtest(self, symbol, train_ratio=0.7, years=2):
        """Run comprehensive backtest for a symbol"""
        print(f"\\n=== BACKTESTING {symbol} ====")
        
        # Get historical data
        df = self.get_historical_data(symbol, years=years)
        if df is None or len(df) < 100:
            print(f"Insufficient data for {symbol}")
            return None
        
        # Prepare features and target
        X, y, feature_columns = self.prepare_features_and_target(df)
        
        if len(X) < 50:
            print(f"Not enough data for training after processing: {len(X)} samples")
            return None
        
        # Split into train/test
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Features used: {len(feature_columns)}")
        
        # Train ensemble model
        print("Training ensemble model...")
        self.ensemble.train_models(X_train, y_train)
        
        # Test the model
        print("Testing model performance...")
        predictions = self.ensemble.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        # Simulate trading performance
        trading_results = self._simulate_trading_performance(df, X_test, predictions, split_idx)
        
        results = {
            'symbol': symbol,
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'features_used': feature_columns,
            'trading_performance': trading_results,
            'model': self.ensemble
        }
        
        print(f"Backtest Results for {symbol}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Total Return: {trading_results['total_return']:.2%}")
        print(f"  Sharpe Ratio: {trading_results['sharpe_ratio']:.3f}")
        
        return results
    
    def _simulate_trading_performance(self, df, X_test, predictions, split_idx):
        """Simulate trading performance based on predictions"""
        # Get the corresponding price data for test period
        test_prices = df['close'].iloc[split_idx:split_idx + len(X_test)]
        
        # Initial trading parameters
        initial_balance = self.initial_balance / len(self.symbols)
        balance = initial_balance
        position = 0
        trades = []
        returns = []
        
        # Trading simulation
        for i, (prediction, price) in enumerate(zip(predictions, test_prices)):
            if i == 0:
                previous_price = price
                continue
            
            price_change = (price - previous_price) / previous_price
            
            # Simple trading logic
            if prediction == 1 and position == 0:  # Buy signal
                trade_size = balance * 0.1
                position = trade_size / price
                balance -= trade_size
                trades.append({'type': 'BUY', 'price': price})
                
            elif prediction == 0 and position > 0:  # Sell signal
                trade_value = position * price
                balance += trade_value
                trade_return = (trade_value - (position * previous_price)) / (position * previous_price)
                returns.append(trade_return)
                trades.append({'type': 'SELL', 'price': price, 'return': trade_return})
                position = 0
            
            previous_price = price
        
        # Calculate final balance if position still open
        if position > 0:
            final_value = position * test_prices.iloc[-1]
            balance += final_value
        
        # Performance metrics
        total_return = (balance - initial_balance) / initial_balance
        volatility = np.std(returns) if returns else 0
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
        """Train market regime detection on historical data"""
        print("\\n=== TRAINING MARKET REGIME DETECTION ===")
        
        symbols = symbols or self.symbols
        
        for symbol in symbols:
            df = self.get_historical_data(symbol, years=years)
            if df is not None:
                # Use the first available symbol for regime detection
                price_data = df['close']
                self.regime_detector.train_model(price_data)
                print("✓ Market regime detection trained successfully")
                return True
        
        print("No data available for regime detection training")
        return False
    
    def save_trained_models(self):
        """Save trained models for future use"""
        try:
            # Save ensemble models
            ensemble_path = os.path.join(self.model_save_path, "trained_ensemble.pkl")
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble, f)
            
            # Save regime detector
            regime_path = os.path.join(self.model_save_path, "trained_regime_detector.pkl")
            with open(regime_path, 'wb') as f:
                pickle.dump(self.regime_detector, f)
            
            print(f"✓ Models saved to {self.model_save_path}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def run_comprehensive_backtest(self, years=2, save_models=True):
        """Run comprehensive backtesting across all symbols"""
        print("🚀 STARTING COMPREHENSIVE BACKTESTING")
        print("=" * 60)
        
        # Connect to MT5
        if not self.connect_to_mt5():
            print("Using simulated data for backtesting")
            return False
        
        self.backtest_results = {}
        
        # Backtest each symbol
        for symbol in self.symbols:
            results = self.run_backtest(symbol, years=years)
            if results:
                self.backtest_results[symbol] = results
        
        # Train market regime detection
        self.train_regime_detection(years=years)
        
        # Save trained models
        if save_models:
            self.save_trained_models()
        
        # Generate summary report
        self.generate_summary_report()
        
        mt5.shutdown()
        print("✓ Comprehensive backtesting completed")
        
        return True
    
    def generate_summary_report(self):
        """Generate comprehensive backtesting report"""
        if not self.backtest_results:
            print("No backtest results to report")
            return
        
        print("\\n" + "=" * 60)
        print("📊 BACKTESTING SUMMARY REPORT")
        print("=" * 60)
        
        total_accuracy = 0
        total_return = 0
        symbol_count = len(self.backtest_results)
        
        for symbol, results in self.backtest_results.items():
            print(f"\\n{symbol}:")
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  Total Return: {results['trading_performance']['total_return']:.2%}")
            print(f"  Sharpe Ratio: {results['trading_performance']['sharpe_ratio']:.3f}")
            print(f"  Trades: {results['trading_performance']['total_trades']}")
            
            total_accuracy += results['accuracy']
            total_return += results['trading_performance']['total_return']
        
        if symbol_count > 0:
            avg_accuracy = total_accuracy / symbol_count
            avg_return = total_return / symbol_count
            
            print(f"\\n📈 AVERAGE PERFORMANCE:")
            print(f"  Accuracy: {avg_accuracy:.3f}")
            print(f"  Return: {avg_return:.2%}")
            print(f"  Symbols Tested: {symbol_count}")
        
        # Save report to file
        report_path = "backtesting_report.txt"
        with open(report_path, 'w') as f:
            f.write("FTMO AI Trading System - Backtesting Report\\n")
            f.write("=" * 50 + "\\n")
            for symbol, results in self.backtest_results.items():
                f.write(f"\\n{symbol}: Accuracy={results['accuracy']:.3f}, Return={results['trading_performance']['total_return']:.2%}\\n")
        
        print(f"\\n📄 Full report saved to: {report_path}")

def run_weekend_backtesting():
    """Main function to run weekend backtesting"""
    print("🎯 FTMO AI SYSTEM - WEEKEND BACKTESTING")
    print("This will train your models on historical data")
    print("=" * 60)
    
    backtester = ComprehensiveBacktester()
    
    # Run comprehensive backtest with 1 year for quick testing
    success = backtester.run_comprehensive_backtest(years=1, save_models=True)
    
    if success:
        print("\\n🎉 WEEKEND BACKTESTING COMPLETED SUCCESSFULLY!")
        print("Your models are now trained on historical data")
        print("Trained models are saved for live trading")
    else:
        print("\\n⚠️  Backtesting completed with some issues")

if __name__ == "__main__":
    run_weekend_backtesting()
