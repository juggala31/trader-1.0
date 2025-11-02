import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import itertools
import sys
import os

class DynamicTimeframeOptimizer:
    def __init__(self):
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4
        }
        
        # Use your actual symbols
        self.symbols = ['US30Z25.sim', 'US100Z25.sim', 'XAUZ25.sim']
        
        # DYNAMIC PARAMETER RANGES
        self.parameter_ranges = {
            'lookback_period': [10, 15, 20, 25, 30, 40, 50],
            'momentum_threshold': [0.5, 0.75, 1.0, 1.25, 1.5],
            'profit_threshold': [0.002, 0.003, 0.005, 0.008, 0.01],
            'stop_loss_ratio': [0.5, 0.6, 0.7, 0.8],
            'confidence_filter': [0.6, 0.65, 0.7, 0.75]
        }
        
        self.results = {}
        
    def connect_mt5(self):
        """Connect to MT5 with better error handling"""
        try:
            if not mt5.initialize():
                print("❌ MT5 initialization failed")
                print("💡 Make sure MT5 is running with your symbols")
                return False
            
            # Check if your symbols are available
            all_symbols = mt5.symbols_get()
            available_symbols = [s.name for s in all_symbols]
            
            print("✅ MT5 connected successfully")
            print(f"📊 Total symbols available: {len(available_symbols)}")
            
            # Filter to find your symbols or similar
            your_symbols = []
            for symbol in self.symbols:
                if symbol in available_symbols:
                    your_symbols.append(symbol)
                    print(f"   ✅ {symbol} is available")
                else:
                    # Find similar symbols
                    similar = [s for s in available_symbols if any(x in s for x in ['US30', 'US100', 'XAU', 'GOLD'])]
                    if similar:
                        print(f"   ❌ {symbol} not found, similar: {similar[:3]}")
                    else:
                        print(f"   ❌ {symbol} not found")
            
            if your_symbols:
                self.symbols = your_symbols
                return True
            else:
                print("❌ None of your symbols are available")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_historical_data(self, symbol, timeframe, bars=1000):
        """Get historical data with robust error handling"""
        try:
            print(f"📥 Fetching {symbol} on {list(self.timeframes.keys())[list(self.timeframes.values()).index(timeframe)]}...")
            
            # Try to get data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                print(f"   ❌ No data returned for {symbol}")
                return None
            
            if len(rates) == 0:
                print(f"   ❌ Empty data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            if len(df) == 0:
                print(f"   ❌ Empty DataFrame for {symbol}")
                return None
            
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            print(f"   ✅ Successfully loaded {len(df)} bars")
            return df
            
        except Exception as e:
            print(f"   ❌ Error fetching {symbol}: {e}")
            return None
    
    def calculate_dynamic_features(self, df, params):
        """Calculate features with FIXED pandas Series handling"""
        if df is None or len(df) < params['lookback_period'] + 10:
            return None
        
        df = df.copy()
        lookback = params['lookback_period']
        
        try:
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['sma_fast'] = df['close'].rolling(5).mean()
            df['sma_slow'] = df['close'].rolling(lookback).mean()
            df['sma_diff'] = (df['sma_fast'] - df['sma_slow']) / df['sma_slow']
            
            # Volatility
            df['volatility'] = df['returns'].rolling(lookback).std()
            
            # Fixed momentum calculation - no ambiguous Series comparisons
            returns = df['close'].pct_change()
            gains = returns.where(returns > 0, 0).rolling(lookback).mean()
            losses = (-returns.where(returns < 0, 0)).rolling(lookback).mean()
            
            # Use vectorized operation instead of conditional
            rs = gains / (gains + losses)
            df['momentum_strength'] = rs.fillna(0.5)  # Fill NaN with neutral value
            
            # Support/resistance
            df['resistance'] = df['high'].rolling(lookback).max()
            df['support'] = df['low'].rolling(lookback).min()
            
            # Safe price position calculation
            range_size = df['resistance'] - df['support']
            df['price_position'] = (df['close'] - df['support']) / range_size
            df['price_position'] = df['price_position'].fillna(0.5)  # Handle division by zero
            
            return df.dropna()
            
        except Exception as e:
            print(f"   ❌ Feature calculation error: {e}")
            return None
    
    def generate_trading_signals(self, df, params):
        """Generate trading signals safely"""
        if df is None or len(df) < 50:
            return None, None
        
        try:
            signals = []
            confidences = []
            
            for i in range(len(df)):
                if i < params['lookback_period']:
                    signals.append(0)
                    confidences.append(0)
                    continue
                
                current = df.iloc[i]
                
                # Use safe attribute access
                momentum = current.get('momentum_strength', 0.5)
                price_pos = current.get('price_position', 0.5)
                volatility = current.get('volatility', 0.01)
                sma_diff = current.get('sma_diff', 0)
                
                signal = 0
                confidence = 0
                
                # Buy conditions
                if (momentum > 0.6 and 
                    price_pos < 0.7 and 
                    volatility < 0.02 and 
                    sma_diff > 0):
                    signal = 1
                    confidence = min(0.9, momentum * 1.2)
                
                # Sell conditions  
                elif (momentum < 0.4 and 
                      price_pos > 0.3 and 
                      volatility < 0.02 and 
                      sma_diff < 0):
                    signal = -1
                    confidence = min(0.9, (1 - momentum) * 1.2)
                
                signals.append(signal)
                confidences.append(confidence)
            
            return signals, confidences
            
        except Exception as e:
            print(f"   ❌ Signal generation error: {e}")
            return None, None
    
    def backtest_with_parameters(self, df, params):
        """Backtest with specific parameters"""
        if df is None or len(df) < 100:
            return 0, 0, 0, 0, "Insufficient data"
        
        df_features = self.calculate_dynamic_features(df, params)
        if df_features is None:
            return 0, 0, 0, 0, "Feature calculation failed"
        
        signals, confidences = self.generate_trading_signals(df_features, params)
        if signals is None:
            return 0, 0, 0, 0, "Signal generation failed"
        
        try:
            # Trading simulation
            capital = 10000
            position = 0
            trades = 0
            wins = 0
            total_pnl = 0
            entry_price = 0
            
            for i in range(len(df_features)):
                if i < params['lookback_period']:
                    continue
                    
                current_price = df_features.iloc[i]['close']
                signal = signals[i]
                confidence = confidences[i]
                
                # Apply confidence filter
                if confidence < params['confidence_filter']:
                    signal = 0
                
                # Trading logic
                if signal != 0 and position == 0:  # Enter trade
                    position = signal
                    entry_price = current_price
                    trades += 1
                    
                elif position != 0:  # Manage position
                    if position == 1:
                        current_pnl = (current_price - entry_price) * 10
                    else:
                        current_pnl = (entry_price - current_price) * 10
                    
                    # Exit conditions
                    profit_target = params['profit_threshold'] * 10000
                    stop_loss = profit_target * params['stop_loss_ratio']
                    
                    exit_trade = False
                    if current_pnl >= profit_target or current_pnl <= -stop_loss:
                        exit_trade = True
                    
                    if exit_trade:
                        capital += current_pnl
                        total_pnl += current_pnl
                        if current_pnl > 0:
                            wins += 1
                        position = 0
            
            # Close final position
            if position != 0:
                final_price = df_features.iloc[-1]['close']
                if position == 1:
                    final_pnl = (final_price - entry_price) * 10
                else:
                    final_pnl = (entry_price - final_price) * 10
                capital += final_pnl
                total_pnl += final_pnl
                if final_pnl > 0:
                    wins += 1
            
            total_return = ((capital - 10000) / 10000) * 100
            win_rate = wins / max(trades, 1)
            profit_factor = total_pnl / max(abs(total_pnl - total_return * 100), 0.1)
            
            return total_return, win_rate, trades, profit_factor, "Success"
            
        except Exception as e:
            print(f"   ❌ Backtesting error: {e}")
            return 0, 0, 0, 0, f"Backtest failed: {e}"
    
    def optimize_parameters_for_timeframe(self, symbol, timeframe_name, timeframe_value, max_combinations=50):
        """Optimize parameters for specific timeframe"""
        print(f"🔧 Optimizing {symbol} on {timeframe_name}...")
        
        df = self.get_historical_data(symbol, timeframe_value, bars=1000)
        if df is None or len(df) < 200:
            print(f"   ❌ Insufficient data for optimization")
            return None, None
        
        tested_combinations = 0
        best_return = -100
        best_params = None
        best_metrics = None
        
        # Test random combinations
        for i in range(max_combinations):
            # Create random parameter set
            params = {}
            for name, values in self.parameter_ranges.items():
                params[name] = np.random.choice(values)
            
            # Backtest with these parameters
            return_pct, win_rate, trades, profit_factor, status = self.backtest_with_parameters(df, params)
            
            if status == "Success" and trades >= 3:
                tested_combinations += 1
                
                # Score this parameter set
                score = return_pct + (win_rate * 50) + (profit_factor * 10)
                
                if score > best_return:
                    best_return = score
                    best_params = params.copy()
                    best_metrics = {
                        'return': return_pct,
                        'win_rate': win_rate,
                        'trades': trades,
                        'profit_factor': profit_factor,
                        'score': score
                    }
            
            # Progress indicator
            if i % 10 == 0:
                print(f"   Progress: {i+1}/{max_combinations} combinations tested")
        
        if best_params:
            print(f"   ✅ Best result: Return={best_metrics['return']:.2f}%, "
                  f"Win Rate={best_metrics['win_rate']:.3f}, Trades={best_metrics['trades']}")
            return best_params, best_metrics
        else:
            print(f"   ❌ No valid results from {tested_combinations} tests")
            return None, None
    
    def run_comprehensive_optimization(self, combinations_per_tf=50):
        """Run optimization across all timeframes"""
        if not self.connect_mt5():
            return False
        
        print("=" * 80)
        print("🚀 FIXED DYNAMIC TIMEFRAME OPTIMIZER")
        print("=" * 80)
        print("Now with proper pandas Series handling")
        print("=" * 80)
        
        for symbol in self.symbols:
            print(f"\n🎯 OPTIMIZING {symbol}")
            print("-" * 50)
            
            symbol_results = {}
            
            for tf_name, tf_value in self.timeframes.items():
                best_params, best_metrics = self.optimize_parameters_for_timeframe(
                    symbol, tf_name, tf_value, combinations_per_tf)
                
                if best_params and best_metrics:
                    symbol_results[tf_name] = {
                        'parameters': best_params,
                        'metrics': best_metrics,
                        'status': 'Success'
                    }
                else:
                    symbol_results[tf_name] = {
                        'parameters': {},
                        'metrics': {'return': 0, 'win_rate': 0, 'trades': 0, 'profit_factor': 0},
                        'status': 'Failed'
                    }
            
            self.results[symbol] = symbol_results
        
        self.generate_optimization_report()
        return True
    
    def generate_optimization_report(self):
        """Generate optimization report"""
        print("\n" + "=" * 80)
        print("📊 OPTIMIZATION RESULTS")
        print("=" * 80)
        
        tf_performance = {}
        
        for symbol in self.symbols:
            print(f"\n🔍 {symbol} RESULTS:")
            print("-" * 50)
            
            valid_results = {tf: res for tf, res in self.results[symbol].items() 
                           if res['status'] == 'Success' and res['metrics']['trades'] >= 3}
            
            if not valid_results:
                print("   No successful optimizations")
                continue
            
            sorted_results = sorted(valid_results.items(), 
                                  key=lambda x: x[1]['metrics']['return'], 
                                  reverse=True)
            
            for tf_name, result in sorted_results:
                metrics = result['metrics']
                print(f"   {tf_name}: Return={metrics['return']:.2f}%, "
                      f"Win Rate={metrics['win_rate']:.3f}, Trades={metrics['trades']}")
                
                if tf_name not in tf_performance:
                    tf_performance[tf_name] = []
                tf_performance[tf_name].append(metrics['return'])
        
        # Overall recommendations
        if tf_performance:
            print("\n🏆 OVERALL RECOMMENDATIONS:")
            avg_returns = {tf: np.mean(returns) for tf, returns in tf_performance.items()}
            ranked_tf = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (tf_name, avg_return) in enumerate(ranked_tf, 1):
                print(f"   {rank}. {tf_name}: Average Return = {avg_return:.2f}%")
            
            best_tf, best_return = ranked_tf[0]
            print(f"\n💡 RECOMMENDATION: Use {best_tf} timeframe")
        
        self.save_results()
    
    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fixed_optimization_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("FIXED DYNAMIC OPTIMIZATION RESULTS\\n")
            f.write("=" * 50 + "\\n")
            
            for symbol in self.symbols:
                f.write(f"{symbol}:\\n")
                for tf_name, result in self.results[symbol].items():
                    f.write(f"  {tf_name}: {result['status']}\\n")
                    if result['status'] == 'Success':
                        f.write(f"    Return: {result['metrics']['return']:.2f}%\\n")
                        f.write(f"    Parameters: {result['parameters']}\\n")
                f.write("\\n")
        
        print(f"\\n💾 Results saved to: {filename}")

# Execute
if __name__ == "__main__":
    print("🚀 Starting FIXED Dynamic Timeframe Optimizer...")
    print("This version fixes the pandas Series comparison error\\n")
    
    optimizer = DynamicTimeframeOptimizer()
    success = optimizer.run_comprehensive_optimization(combinations_per_tf=50)
    
    if success:
        print("\\n✅ Optimization completed successfully!")
    else:
        print("\\n❌ Optimization failed")
