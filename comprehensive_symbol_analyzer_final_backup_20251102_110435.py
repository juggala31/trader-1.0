import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveSymbolAnalyzer:
    def __init__(self):
        # Your actual OANDA symbols
        self.symbols = [
            # Indices
            'US30Z25.sim', 'US100Z25.sim', 'US500Z25.sim',
            # Precious Metals
            'XAUZ25.sim',
            # Commodities
            'USOILZ25.sim',
            # Cryptocurrencies
            'BTCX25.sim'
        ]
        
        # All timeframes to test
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4
        }
        
        # Dynamic parameter ranges for optimization
        self.parameter_ranges = {
            'lookback_period': [5, 10, 15, 20, 25, 30, 40, 50],
            'momentum_threshold': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            'profit_threshold': [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015],
            'stop_loss_ratio': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            'confidence_filter': [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        }
        
        self.results = {}
        
    def connect_mt5(self):
        """Connect to MT5"""
        try:
            if not mt5.initialize():
                print("❌ MT5 initialization failed")
                return False
            
            print("✅ MT5 connected successfully")
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_historical_data(self, symbol, timeframe, bars=2000):
        """Get historical data for analysis"""
        try:
            print(f"📥 Fetching {symbol} on {list(self.timeframes.keys())[list(self.timeframes.values()).index(timeframe)]}...")
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                print(f"   ❌ No data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            if len(df) < 100:
                print(f"   ❌ Insufficient data: {len(df)} bars")
                return None
            
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Calculate actual date range
            actual_start = df.index.min()
            actual_end = df.index.max()
            days_covered = (actual_end - actual_start).days
            
            print(f"   ✅ Loaded {len(df):,} bars covering {days_covered} days")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Error fetching {symbol}: {e}")
            return None
    
    def calculate_dynamic_features(self, df, params):
        """Calculate features with dynamic parameters"""
        if df is None or len(df) < params['lookback_period'] + 10:
            return None
        
        df = df.copy()
        lookback = params['lookback_period']
        
        try:
            # Price features
            df['returns'] = df['close'].pct_change()
            df['sma_fast'] = df['close'].rolling(max(5, lookback//3)).mean()
            df['sma_slow'] = df['close'].rolling(lookback).mean()
            df['sma_diff'] = (df['sma_fast'] - df['sma_slow']) / df['sma_slow']
            
            # Volatility
            df['volatility'] = df['returns'].rolling(lookback).std()
            
            # Momentum indicators
            returns = df['close'].pct_change()
            gains = returns.where(returns > 0, 0).rolling(lookback).mean()
            losses = (-returns.where(returns < 0, 0)).rolling(lookback).mean()
            rs = gains / (gains + losses)
            df['momentum_strength'] = rs.fillna(0.5)
            
            # Support/resistance
            df['resistance'] = df['high'].rolling(lookback).max()
            df['support'] = df['low'].rolling(lookback).min()
            range_size = df['resistance'] - df['support']
            df['price_position'] = (df['close'] - df['support']) / range_size
            df['price_position'] = df['price_position'].fillna(0.5)
            
            return df.dropna()
            
        except Exception as e:
            print(f"   ❌ Feature calculation error: {e}")
            return None
    
    def generate_trading_signals(self, df, params):
        """Generate trading signals with dynamic parameters"""
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
                
                momentum = current.get('momentum_strength', 0.5)
                price_pos = current.get('price_position', 0.5)
                volatility = current.get('volatility', 0.01)
                sma_diff = current.get('sma_diff', 0)
                
                signal = 0
                confidence = 0
                
                # Dynamic signal generation based on parameters
                if (momentum > params['momentum_threshold'] and 
                    price_pos < 0.7 and 
                    volatility < 0.02 and 
                    sma_diff > 0):
                    signal = 1
                    confidence = min(0.9, momentum * 1.2)
                
                elif (momentum < (2 - params['momentum_threshold']) and 
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
            entry_idx = 0
            
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
                    entry_idx = i
                    trades += 1
                    
                elif position != 0:  # Manage position
                    if position == 1:
                        current_pnl = (current_price - entry_price) * 10
                    else:
                        current_pnl = (entry_price - current_price) * 10
                    
                    # Dynamic exit conditions
                    profit_target = params['profit_threshold'] * 10000
                    stop_loss = profit_target * params['stop_loss_ratio']
                    
                    exit_trade = False
                    if current_pnl >= profit_target or current_pnl <= -stop_loss:
                        exit_trade = True
                    
                    # Time-based exit
                    if i - entry_idx > 100:  # Max 100 bars
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
    
    def optimize_parameters_for_symbol_timeframe(self, symbol, timeframe_name, timeframe_value, max_combinations=80):
        """Optimize parameters for specific symbol and timeframe"""
        print(f"🔧 Optimizing {symbol} on {timeframe_name}...")
        
        df = self.get_historical_data(symbol, timeframe_value, bars=2000)
        if df is None or len(df) < 200:
            print(f"   ❌ Insufficient data for optimization")
            return None, None
        
        tested_combinations = 0
        best_score = -1000
        best_params = None
        best_metrics = None
        
        # Test random parameter combinations
        for i in range(max_combinations):
            # Create random parameter set
            params = {}
            for name, values in self.parameter_ranges.items():
                params[name] = np.random.choice(values)
            
            # Backtest with these parameters
            return_pct, win_rate, trades, profit_factor, status = self.backtest_with_parameters(df, params)
            
            if status == "Success" and trades >= 5:
                tested_combinations += 1
                
                # Comprehensive scoring
                score = (return_pct * 2) + (win_rate * 100) + (profit_factor * 50) + (trades * 0.1)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = {
                        'return': return_pct,
                        'win_rate': win_rate,
                        'trades': trades,
                        'profit_factor': profit_factor,
                        'score': score
                    }
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{max_combinations} combinations tested")
        
        if best_params:
            print(f"   ✅ Best: Return={best_metrics['return']:.2f}%, "
                  f"Win Rate={best_metrics['win_rate']:.3f}, Trades={best_metrics['trades']}, "
                  f"PF={best_metrics['profit_factor']:.2f}")
            return best_params, best_metrics
        else:
            print(f"   ❌ No valid results from {tested_combinations} tests")
            return None, None
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis across all symbols and timeframes"""
        if not self.connect_mt5():
            return False
        
        print("=" * 80)
        print("🚀 COMPREHENSIVE SYMBOL & TIMEFRAME ANALYSIS")
        print("=" * 80)
        print("Testing ALL symbols across M5, M15, M30, H1, H4 timeframes")
        print("With dynamic parameter optimization (80 combinations each)")
        print("=" * 80)
        
        total_tests = len(self.symbols) * len(self.timeframes) * 80
        print(f"Total parameter tests: {total_tests:,}")
        print("=" * 80)
        
        for symbol in self.symbols:
            print(f"\n🎯 ANALYZING {symbol}")
            print("-" * 50)
            
            symbol_results = {}
            
            for tf_name, tf_value in self.timeframes.items():
                best_params, best_metrics = self.optimize_parameters_for_symbol_timeframe(
                    symbol, tf_name, tf_value, 80)
                
                if best_params and best_metrics:
                    symbol_results[tf_name] = {
                        'parameters': best_params,
                        'metrics': best_metrics,
                        'status': 'Success'
                    }
                else:
                    symbol_results[tf_name] = {
                        'parameters': {},
                        'metrics': {'return': 0, 'win_rate': 0, 'trades': 0, 'profit_factor': 0, 'score': -1000},
                        'status': 'Failed'
                    }
            
            self.results[symbol] = symbol_results
        
        self.generate_comprehensive_report()
        return True
    
    def generate_comprehensive_report(self):
        """Generate detailed analysis report"""
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE SYMBOL & TIMEFRAME ANALYSIS REPORT")
        print("=" * 100)
        
        # Collect all results for ranking
        all_results = []
        
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
                      f"Profit={params['profit_threshold']:.3f}")
                
                # Store for overall ranking
                all_results.append({
                    'symbol': symbol,
                    'timeframe': tf_name,
                    'return': metrics['return'],
                    'win_rate': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'score': metrics['score'],
                    'parameters': params
                })
        
        # Overall rankings
        if all_results:
            print("\n" + "=" * 100)
            print("🏆 TOP PERFORMING SYMBOL-TIMEFRAME COMBINATIONS")
            print("=" * 100)
            
            # Sort by return
            ranked_results = sorted(all_results, key=lambda x: x['return'], reverse=True)
            
            print("📈 RANKED BY RETURN:")
            for i, result in enumerate(ranked_results[:10], 1):
                print(f"{i:2d}. {result['symbol']:10} ({result['timeframe']:3}) | "
                      f"Return: {result['return']:6.2f}% | "
                      f"Win Rate: {result['win_rate']:.3f} | "
                      f"PF: {result['profit_factor']:.2f}")
            
            # Best overall recommendation
            best_overall = ranked_results[0]
            print(f"\n💡 **TOP RECOMMENDATION**: {best_overall['symbol']} on {best_overall['timeframe']}")
            print(f"   Return: {best_overall['return']:.2f}%")
            print(f"   Win Rate: {best_overall['win_rate']:.3f}")
            print(f"   Profit Factor: {best_overall['profit_factor']:.2f}")
            
            # Best by timeframe
            print(f"\n🎯 BEST BY TIMEFRAME:")
            for tf in self.timeframes.keys():
                tf_results = [r for r in all_results if r['timeframe'] == tf]
                if tf_results:
                    best_tf = max(tf_results, key=lambda x: x['return'])
                    print(f"   {tf}: {best_tf['symbol']} ({best_tf['return']:.2f}%)")
        
        self.save_detailed_results()
    
    def save_detailed_results(self):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_analysis_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("COMPREHENSIVE SYMBOL & TIMEFRAME ANALYSIS\\n")
            f.write("=" * 70 + "\\n")
            f.write(f"Analysis Date: {datetime.now()}\\n")
            f.write("Broker: OANDA MT5\\n")
            f.write("Symbols: Indices + Metals + Crypto + Oil\\n")
            f.write("Timeframes: M5, M15, M30, H1, H4\\n\\n")
            
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
                        f.write(f"  Score: {metrics['score']:.1f}\\n")
                        f.write(f"  Parameters: {params}\\n")
                f.write("\\n")
        
        print(f"\\n💾 Full results saved to: {filename}")

# Execute comprehensive analysis
if __name__ == "__main__":
    print("🚀 Starting Comprehensive Symbol & Timeframe Analysis...")
    print("Testing ALL your symbols with dynamic parameter optimization")
    print("This will find the BEST combinations for your AI trading system\\n")
    
    analyzer = ComprehensiveSymbolAnalyzer()
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("\\n✅ Analysis completed successfully!")
        print("Check the report for data-driven symbol and timeframe recommendations")
    else:
        print("\\n❌ Analysis failed")
