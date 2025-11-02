import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeBacktester:
    def __init__(self):
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15, 
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4
        }
        self.instruments = ['US30', 'US100', 'XAUUSD']  # Using standard symbols
        self.results = {}
        
    def connect_mt5(self):
        if not mt5.initialize():
            print("MT5 initialization failed")
            return False
        return True
    
    def get_historical_data(self, symbol, timeframe, bars=1000):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None:
            print(f"No data for {symbol} on this timeframe")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    
    def calculate_features(self, df):
        if df is None or len(df) < 100:
            return None
            
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Target: Next bar direction
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df.dropna()
    
    def train_model(self, df):
        if df is None or len(df) < 100:
            return None, 0, [], None, None, None
            
        feature_cols = [col for col in df.columns if col not in ['target', 'tick_volume', 'real_volume', 'spread']]
        
        X = df[feature_cols]
        y = df['target']
        
        if len(X) < 50:
            return None, 0, [], None, None, None
            
        split_point = int(len(X) * 0.7)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, accuracy, feature_cols, X_test, y_test, y_pred
    
    def simulate_trading(self, df, model, feature_cols):
        if df is None or len(df) < 100 or model is None:
            return 0, 0, 0
            
        test_size = min(300, len(df) // 2)
        test_df = df.tail(test_size)
        
        if len(test_df) < 50:
            return 0, 0, 0
            
        X_test = test_df[feature_cols]
        signals = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        capital = 10000
        position = 0
        trades = 0
        wins = 0
        
        for i in range(1, len(test_df)):
            current_price = test_df.iloc[i]['close']
            prev_price = test_df.iloc[i-1]['close']
            
            # Simple strategy: trade on high confidence signals
            if probabilities[i] > 0.65:  # High confidence threshold
                if signals[i] == 1 and position <= 0:  # Buy
                    if position < 0:  # Close short
                        pnl = (entry_price - current_price) * 10
                        capital += pnl
                        trades += 1
                        if pnl > 0: wins += 1
                    
                    position = 1
                    entry_price = current_price
                    
                elif signals[i] == 0 and position >= 0:  # Sell
                    if position > 0:  # Close long
                        pnl = (current_price - entry_price) * 10
                        capital += pnl
                        trades += 1
                        if pnl > 0: wins += 1
                    
                    position = -1
                    entry_price = current_price
        
        # Close final position
        if position != 0:
            current_price = test_df.iloc[-1]['close']
            if position > 0:
                pnl = (current_price - entry_price) * 10
            else:
                pnl = (entry_price - current_price) * 10
            capital += pnl
            trades += 1
            if pnl > 0: wins += 1
        
        total_return = ((capital - 10000) / 10000) * 100
        win_rate = wins / max(trades, 1)
        
        return total_return, win_rate, trades
    
    def run_backtest(self, symbol):
        symbol_results = {}
        
        for tf_name, tf_value in self.timeframes.items():
            print(f"Testing {symbol} on {tf_name}...")
            
            df = self.get_historical_data(symbol, tf_value, bars=1500)
            if df is None:
                symbol_results[tf_name] = {'accuracy': 0, 'return': 0, 'win_rate': 0, 'trades': 0, 'status': 'No data'}
                continue
                
            df_features = self.calculate_features(df)
            if df_features is None:
                symbol_results[tf_name] = {'accuracy': 0, 'return': 0, 'win_rate': 0, 'trades': 0, 'status': 'Insufficient data'}
                continue
            
            model, accuracy, feature_cols, X_test, y_test, y_pred = self.train_model(df_features)
            
            if model is None:
                symbol_results[tf_name] = {'accuracy': 0, 'return': 0, 'win_rate': 0, 'trades': 0, 'status': 'Model failed'}
                continue
            
            total_return, win_rate, num_trades = self.simulate_trading(df_features, model, feature_cols)
            
            symbol_results[tf_name] = {
                'accuracy': accuracy,
                'return': total_return,
                'win_rate': win_rate,
                'trades': num_trades,
                'status': 'Success'
            }
            
            print(f"  {tf_name}: Accuracy={accuracy:.3f}, Return={total_return:.2f}%, Trades={num_trades}")
        
        return symbol_results
    
    def run_comprehensive_backtest(self):
        if not self.connect_mt5():
            return False
            
        print("=" * 60)
        print("MULTI-TIMEFRAME BACKTESTING ANALYSIS")
        print("=" * 60)
        
        for symbol in self.instruments:
            print(f"\n📊 BACKTESTING {symbol}")
            print("-" * 40)
            
            symbol_results = self.run_backtest(symbol)
            self.results[symbol] = symbol_results
            
        self.generate_report()
        mt5.shutdown()
        return True
    
    def generate_report(self):
        print("\n" + "=" * 80)
        print("📈 COMPREHENSIVE BACKTESTING REPORT")
        print("=" * 80)
        
        timeframe_performance = {}
        
        for symbol in self.instruments:
            print(f"\n🎯 {symbol} RESULTS:")
            print("-" * 50)
            
            valid_results = {tf: res for tf, res in self.results[symbol].items() if res['status'] == 'Success'}
            
            if not valid_results:
                print("No successful backtests for this symbol")
                continue
                
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['return'], reverse=True)
            
            for tf_name, results in sorted_results:
                print(f"{tf_name:>4} | Accuracy: {results['accuracy']:.3f} | "
                      f"Return: {results['return']:7.2f}% | "
                      f"Win Rate: {results['win_rate']:.3f} | "
                      f"Trades: {results['trades']:3d}")
                
                if tf_name not in timeframe_performance:
                    timeframe_performance[tf_name] = []
                timeframe_performance[tf_name].append(results['return'])
        
        # Overall rankings
        print("\n" + "=" * 80)
        print("🏆 OVERALL TIMEFRAME PERFORMANCE")
        print("=" * 80)
        
        avg_returns = {}
        for tf_name, returns in timeframe_performance.items():
            if returns:  # Only include timeframes with valid results
                avg_returns[tf_name] = np.mean(returns)
        
        if avg_returns:
            ranked_timeframes = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (tf_name, avg_return) in enumerate(ranked_timeframes, 1):
                print(f"{rank:2d}. {tf_name}: Average Return = {avg_return:.2f}%")
            
            best_tf, best_return = ranked_timeframes[0]
            print(f"\n💡 RECOMMENDATION: Use {best_tf} timeframe")
            print(f"   Average return: {best_return:.2f}%")
        else:
            print("No valid backtest results available")
        
        self.save_results()
    
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timeframe_analysis_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("MULTI-TIMEFRAME ANALYSIS RESULTS\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"Generated: {datetime.now()}\\n\\n")
            
            for symbol in self.instruments:
                f.write(f"{symbol} RESULTS:\\n")
                f.write("-" * 30 + "\\n")
                
                for tf_name, results in self.results[symbol].items():
                    f.write(f"{tf_name}: {results['status']}, ")
                    if results['status'] == 'Success':
                        f.write(f"Accuracy={results['accuracy']:.3f}, Return={results['return']:.2f}%, ")
                        f.write(f"WinRate={results['win_rate']:.3f}, Trades={results['trades']}\\n")
                    else:
                        f.write("\\n")
                f.write("\\n")
        
        print(f"\\n💾 Results saved to: {filename}")

# Execute
if __name__ == "__main__":
    print("Starting Multi-Timeframe Backtesting Analysis...")
    backtester = MultiTimeframeBacktester()
    success = backtester.run_comprehensive_backtest()
    
    if success:
        print("\\n✅ Analysis completed successfully!")
    else:
        print("\\n❌ Analysis failed!")
