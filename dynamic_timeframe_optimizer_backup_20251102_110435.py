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
        
        self.symbols = ['US30', 'US100', 'XAUUSD']  # Standard symbols that should work
        
        # DYNAMIC PARAMETER RANGES (like your 3000+ combinations)
        self.parameter_ranges = {
            'lookback_period': [5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
            'momentum_threshold': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
            'profit_threshold': [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02],
            'stop_loss_ratio': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            'confidence_filter': [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        }
        
        self.results = {}
        self.best_parameters = {}
        
    def connect_mt5(self):
        """Connect to MT5 with robust error handling"""
        try:
            if not mt5.initialize():
                print("❌ MT5 initialization failed")
                print("💡 Make sure MT5 is running with demo account")
                return False
            
            # Test connection
            symbols = mt5.symbols_get()
            if not symbols:
                print("❌ No symbols available - check MT5 connection")
                return False
                
            print("✅ MT5 connected successfully")
            
            # Show available symbols for debugging
            available_symbols = [s.name for s in symbols if any(x in s.name for x in ['US30', 'US100', 'XAU', 'GOLD'])]
            print(f"📊 Available symbols: {available_symbols}")
            
            # Update symbols to available ones
            if available_symbols:
                self.symbols = available_symbols[:3]  # Use first 3 available
                print(f"🎯 Using symbols: {self.symbols}")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_historical_data(self, symbol, timeframe, bars=1000):
        """Robust data fetching with fallbacks"""
        try:
            print(f"📥 Fetching {symbol} on {timeframe}...")
            
            # Try multiple methods to get data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                # Try alternative method
                rates = mt5.copy_rates_range(symbol, timeframe, 
                                           datetime.now() - timedelta(days=30), 
                                           datetime.now())
            
            if rates is None or len(rates) == 0:
                print(f"   ❌ No data for {symbol} on this timeframe")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            print(f"   ✅ Got {len(df)} bars")
            return df
            
        except Exception as e:
            print(f"   ❌ Error fetching data: {e}")
            return None
    
    def calculate_dynamic_features(self, df, params):
        """Calculate features using dynamic parameters"""
        if df is None or len(df) < params['lookback_period'] + 10:
            return None
        
        df = df.copy()
        
        # Price features with dynamic lookback
        lookback = params['lookback_period']
        df['sma_fast'] = df['close'].rolling(max(5, lookback//3)).mean()
        df['sma_slow'] = df['close'].rolling(lookback).mean()
        df['price_momentum'] = (df['close'] - df['close'].shift(lookback)) / df['close'].shift(lookback)
        
        # Volatility with dynamic period
        df['volatility'] = df['close'].pct_change().rolling(lookback).std()
        
        # RSI-like momentum indicator
        returns = df['close'].pct_change()
        gains = returns.where(returns > 0, 0).rolling(lookback).mean()
        losses = (-returns.where(returns < 0, 0)).rolling(lookback).mean()
        df['momentum_strength'] = gains / (gains + losses) if (gains + losses) > 0 else 0.5
        
        # Support/resistance levels
        df['resistance'] = df['high'].rolling(lookback).max()
        df['support'] = df['low'].rolling(lookback).min()
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        return df.dropna()
    
    def generate_trading_signals(self, df, params):
        """Generate signals using dynamic parameters"""
        if df is None or len(df) < 50:
            return None, None
        
        signals = []
        confidences = []
        
        for i in range(len(df)):
            if i < params['lookback_period']:
                signals.append(0)
                confidences.append(0)
                continue
            
            current = df.iloc[i]
            momentum = current['momentum_strength']
            price_pos = current['price_position']
            volatility = current['volatility']
            
            # Dynamic signal generation
            signal = 0
            confidence = 0
            
            # Buy signal conditions
            if (momentum > 0.6 and 
                price_pos < 0.7 and 
                volatility < 0.02 and 
                current['sma_fast'] > current['sma_slow']):
                signal = 1
                confidence = min(0.9, momentum * 1.5)
            
            # Sell signal conditions  
            elif (momentum < 0.4 and 
                  price_pos > 0.3 and 
                  volatility < 0.02 and 
                  current['sma_fast'] < current['sma_slow']):
                signal = -1
                confidence = min(0.9, (1 - momentum) * 1.5)
            
            signals.append(signal)
            confidences.append(confidence)
        
        return signals, confidences
    
    def backtest_with_parameters(self, df, params):
        """Backtest with specific parameter set"""
        if df is None or len(df) < 100:
            return 0, 0, 0, 0, "Insufficient data"
        
        df_features = self.calculate_dynamic_features(df, params)
        if df_features is None:
            return 0, 0, 0, 0, "Feature calculation failed"
        
        signals, confidences = self.generate_trading_signals(df_features, params)
        if signals is None:
            return 0, 0, 0, 0, "Signal generation failed"
        
        # Trading simulation
        capital = 10000
        position = 0
        trades = 0
        wins = 0
        total_pnl = 0
        entry_price = 0
        entry_idx = 0
        
        for i in range(len(df_features)):
            if i < params['lookback_period']:
                continue
                
            current_price = df_features.iloc[i]['close']
            signal = signals[i]
            confidence = confidences[i]
            
            # Only trade if confidence meets threshold
            if confidence < params['confidence_filter']:
                signal = 0
            
            # Trading logic
            if signal != 0 and position == 0:  # Enter trade
                position = signal
                entry_price = current_price
                entry_idx = i
                trades += 1
                
            elif position != 0:  # Manage existing position
                # Calculate current PnL
                if position == 1:
                    current_pnl = (current_price - entry_price) * 100
                else:
                    current_pnl = (entry_price - current_price) * 100
                
                # Check exit conditions
                exit_trade = False
                
                # Profit target
                if abs(current_pnl) >= params['profit_threshold'] * 10000:
                    exit_trade = True
                
                # Stop loss (dynamic based on profit threshold)
                stop_loss = params['profit_threshold'] * params['stop_loss_ratio'] * 10000
                if current_pnl <= -stop_loss:
                    exit_trade = True
                
                # Time-based exit (prevent holding too long)
                if i - entry_idx > 50:  # Max 50 bars
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
                final_pnl = (final_price - entry_price) * 100
            else:
                final_pnl = (entry_price - final_price) * 100
            capital += final_pnl
            total_pnl += final_pnl
            if final_pnl > 0:
                wins += 1
        
        total_return = ((capital - 10000) / 10000) * 100
        win_rate = wins / max(trades, 1)
        profit_factor = total_pnl / max(abs(total_pnl - total_return * 100), 1)
        
        return total_return, win_rate, trades, profit_factor, "Success"
    
    def optimize_parameters_for_timeframe(self, symbol, timeframe_name, timeframe_value, max_combinations=100):
        """Optimize parameters for specific timeframe"""
        print(f"🔧 Optimizing {symbol} on {timeframe_name}...")
        
        # Get data first
        df = self.get_historical_data(symbol, timeframe_value, bars=1500)
        if df is None or len(df) < 200:
            print(f"   ❌ Insufficient data for optimization")
            return None
        
        # Generate parameter combinations (sample from full space)
        param_names = list(self.parameter_ranges.keys())
        param_values = list(self.parameter_ranges.values())
        
        # Use sampling to test diverse combinations
        tested_combinations = 0
        best_return = -100
        best_params = None
        best_metrics = None
        
        # Test random combinations for efficiency
        for _ in range(max_combinations):
            # Create random parameter set
            params = {}
            for name, values in self.parameter_ranges.items():
                params[name] = np.random.choice(values)
            
            # Backtest with these parameters
            return_pct, win_rate, trades, profit_factor, status = self.backtest_with_parameters(df, params)
            
            if status == "Success" and trades >= 5:  # Valid result
                tested_combinations += 1
                
                # Score this parameter set (weighted combination)
                score = return_pct + (win_rate * 50) + (profit_factor * 20)
                
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
        
        if best_params:
            print(f"   ✅ Best: Return={best_metrics['return']:.2f}%, Win Rate={best_metrics['win_rate']:.3f}, "
                  f"Trades={best_metrics['trades']}, PF={best_metrics['profit_factor']:.2f}")
            return best_params, best_metrics
        else:
            print(f"   ❌ No valid parameter sets found")
            return None, None
    
    def run_comprehensive_optimization(self, combinations_per_tf=50):
        """Run optimization across all timeframes and symbols"""
        if not self.connect_mt5():
            return False
        
        print("=" * 80)
        print("🚀 DYNAMIC TIMEFRAME & PARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"Testing up to {combinations_per_tf} parameter combinations per timeframe")
        print(f"Total tests: {len(self.symbols)} symbols × {len(self.timeframes)} TFs × {combinations_per_tf} combinations")
        print("=" * 80)
        
        overall_results = {}
        
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
                        'metrics': {'return': 0, 'win_rate': 0, 'trades': 0, 'profit_factor': 0, 'score': -100},
                        'status': 'Failed'
                    }
            
            self.results[symbol] = symbol_results
            overall_results[symbol] = symbol_results
        
        self.generate_optimization_report()
        return True
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        print("\n" + "=" * 100)
        print("📊 DYNAMIC OPTIMIZATION RESULTS")
        print("=" * 100)
        
        # Calculate overall best timeframe
        tf_scores = {}
        tf_returns = {}
        
        for symbol in self.symbols:
            print(f"\n🔍 {symbol} - OPTIMIZED PERFORMANCE:")
            print("-" * 70)
            
            valid_results = {tf: res for tf, res in self.results[symbol].items() 
                           if res['status'] == 'Success' and res['metrics']['trades'] >= 5}
            
            if not valid_results:
                print("   No successful optimizations")
                continue
            
            # Sort by return
            sorted_results = sorted(valid_results.items(), 
                                  key=lambda x: x[1]['metrics']['return'], 
                                  reverse=True)
            
            for tf_name, result in sorted_results:
                metrics = result['metrics']
                params = result['parameters']
                
                print(f"   {tf_name:>4} | Return: {metrics['return']:7.2f}% | "
                      f"Win Rate: {metrics['win_rate']:.3f} | "
                      f"Trades: {metrics['trades']:3d} | "
                      f"Profit Factor: {metrics['profit_factor']:.2f}")
                
                print(f"        Parameters: LB={params['lookback_period']}, "
                      f"Thresh={params['momentum_threshold']}, "
                      f"Profit={params['profit_threshold']:.3f}, "
                      f"SL Ratio={params['stop_loss_ratio']}, "
                      f"Conf={params['confidence_filter']}")
                
                # Aggregate scores
                if tf_name not in tf_scores:
                    tf_scores[tf_name] = []
                    tf_returns[tf_name] = []
                
                tf_scores[tf_name].append(metrics['score'])
                tf_returns[tf_name].append(metrics['return'])
        
        # Overall recommendations
        print("\n" + "=" * 100)
        print("🏆 OVERALL OPTIMIZATION RECOMMENDATIONS")
        print("=" * 100)
        
        if tf_scores:
            # Calculate average performance
            avg_returns = {tf: np.mean(returns) for tf, returns in tf_returns.items() if returns}
            avg_scores = {tf: np.mean(scores) for tf, scores in tf_scores.items() if scores}
            
            # Rank by average return
            ranked_tf = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)
            
            print("📈 PERFORMANCE RANKING (by average return):")
            for rank, (tf_name, avg_return) in enumerate(ranked_tf, 1):
                avg_score = avg_scores[tf_name]
                print(f"   {rank:2d}. {tf_name}: Return = {avg_return:.2f}%, Score = {avg_score:.2f}")
            
            # Best timeframe recommendation
            best_tf, best_return = ranked_tf[0]
            best_score = avg_scores[best_tf]
            
            print(f"\n💡 **TOP RECOMMENDATION**: {best_tf} Timeframe")
            print(f"   Average Return: {best_return:.2f}%")
            print(f"   Overall Score: {best_score:.2f}")
            
            # Show best parameters for top timeframe
            print(f"\n🎯 OPTIMAL PARAMETERS for {best_tf}:")
            for symbol in self.symbols:
                if best_tf in self.results[symbol] and self.results[symbol][best_tf]['status'] == 'Success':
                    params = self.results[symbol][best_tf]['parameters']
                    metrics = self.results[symbol][best_tf]['metrics']
                    print(f"   {symbol}: Return={metrics['return']:.2f}% with "
                          f"LB={params['lookback_period']}, Thresh={params['momentum_threshold']}")
        
        self.save_detailed_optimization_results()
    
    def save_detailed_optimization_results(self):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dynamic_optimization_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("DYNAMIC TIMEFRAME PARAMETER OPTIMIZATION RESULTS\\n")
            f.write("=" * 70 + "\\n")
            f.write(f"Optimization Date: {datetime.now()}\\n")
            f.write(f"Parameter Space: {sum(len(v) for v in self.parameter_ranges.values())} total combinations\\n\\n")
            
            for symbol in self.symbols:
                f.write(f"SYMBOL: {symbol}\\n")
                f.write("-" * 50 + "\\n")
                
                for tf_name, result in self.results[symbol].items():
                    f.write(f"{tf_name}: {result['status']}\\n")
                    if result['status'] == 'Success':
                        metrics = result['metrics']
                        params = result['parameters']
                        f.write(f"  Return: {metrics['return']:.2f}%\\n")
                        f.write(f"  Win Rate: {metrics['win_rate']:.3f}\\n")
                        f.write(f"  Trades: {metrics['trades']}\\n")
                        f.write(f"  Profit Factor: {metrics['profit_factor']:.2f}\\n")
                        f.write(f"  Parameters: {params}\\n")
                f.write("\\n")
        
        print(f"\\n💾 Full results saved to: {filename}")

# Execute optimization
if __name__ == "__main__":
    print("🚀 Starting Dynamic Timeframe Parameter Optimization...")
    print("This will test thousands of parameter combinations across all timeframes")
    print("Similar to your previous 3000+ parameter testing approach\\n")
    
    optimizer = DynamicTimeframeOptimizer()
    success = optimizer.run_comprehensive_optimization(combinations_per_tf=80)  # Test 80 combos per TF
    
    if success:
        print("\\n✅ Optimization completed successfully!")
        print("Check the report for data-driven timeframe recommendations")
    else:
        print("\\n❌ Optimization failed")
