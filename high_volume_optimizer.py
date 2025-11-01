import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import warnings
from itertools import product
import time
warnings.filterwarnings("ignore")

class HighVolumeOptimizer:
    def __init__(self, symbols=None, initial_balance=100000):
        self.symbols = symbols or ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]
        self.initial_balance = initial_balance
        self.optimization_results = {}
        self.best_parameters = {}
        self.model_save_path = "optimized_models/"
        self.progress_callback = None
        
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def set_progress_callback(self, callback):
        self.progress_callback = callback
    
    def log_progress(self, message):
        print(message)
        if self.progress_callback:
            self.progress_callback(message)
    
    # EXPANDED PARAMETER SPACES FOR 3000+ COMBINATIONS
    def get_extended_parameter_spaces(self):
        """Extended parameter spaces for 3000+ combinations"""
        return {
            'timeframe': {
                'options': ['D1', 'H4', 'H1'],
                'description': 'Chart timeframe'
            },
            'feature_set': {
                'options': ['basic', 'standard', 'advanced', 'comprehensive', 'minimal', 'technical_only'],
                'description': 'Technical indicator complexity'
            },
            'lookback_periods': {
                'options': [
                    [5, 10, 20],      # Short-term
                    [10, 20, 50],     # Medium-term  
                    [20, 50, 100],    # Long-term
                    [5, 20, 50],      # Mixed
                    [7, 14, 21],      # Fibonacci-based
                    [8, 21, 34],      # Alternative periods
                    [3, 9, 27]        # Geometric progression
                ],
                'description': 'Moving average periods'
            },
            'volatility_window': {
                'options': [5, 10, 14, 20, 30, 50, 100],
                'description': 'Volatility calculation window'
            },
            'rsi_period': {
                'options': [7, 9, 14, 21, 28],
                'description': 'RSI period'
            },
            'macd_fast': {
                'options': [8, 12, 16],
                'description': 'MACD fast period'
            },
            'macd_slow': {
                'options': [21, 26, 30],
                'description': 'MACD slow period'
            },
            'target_horizon': {
                'options': [1, 2, 3, 5, 7, 10],
                'description': 'Prediction horizon (days)'
            },
            'train_ratio': {
                'options': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
                'description': 'Train/test split ratio'
            },
            'price_sensitivity': {
                'options': ['low', 'medium', 'high', 'adaptive'],
                'description': 'Price movement sensitivity'
            }
        }
    
    def generate_3000_combinations(self):
        """Generate 3000+ parameter combinations"""
        spaces = self.get_extended_parameter_spaces()
        
        # Create grid of parameters
        param_grid = {}
        for param_name, param_info in spaces.items():
            param_grid[param_name] = param_info['options']
        
        # Generate all possible combinations
        all_combinations = list(product(*param_grid.values()))
        self.log_progress(f"Total possible combinations: {len(all_combinations):,}")
        
        # If more than 3000, use intelligent sampling
        if len(all_combinations) > 3000:
            # Use stratified sampling to ensure diversity
            step_size = max(1, len(all_combinations) // 3000)
            combinations = all_combinations[::step_size][:3000]
            self.log_progress(f"Sampled to {len(combinations)} combinations")
        else:
            combinations = all_combinations
        
        # Convert to parameter dictionaries
        param_dicts = []
        for combo in combinations:
            param_dict = {}
            for i, param_name in enumerate(param_grid.keys()):
                param_dict[param_name] = combo[i]
            param_dicts.append(param_dict)
        
        self.log_progress(f"🎯 Testing {len(param_dicts):,} parameter combinations")
        return param_dicts
    
    def get_historical_data_extended(self, symbol, years=5, timeframe='D1'):
        """Get extended historical data for thorough testing"""
        try:
            # Map timeframe to MT5 constant
            timeframe_map = {'D1': mt5.TIMEFRAME_D1, 'H4': mt5.TIMEFRAME_H4, 'H1': mt5.TIMEFRAME_H1}
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_D1)
            
            # Get more data for better testing
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is not None and len(rates) > 200:  # Require more data for thorough testing
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return df
            else:
                return self.generate_extended_data(symbol, years, timeframe)
                
        except Exception as e:
            self.log_progress(f"Data error: {e}")
            return self.generate_extended_data(symbol, years, timeframe)
    
    def generate_extended_data(self, symbol, years, timeframe):
        """Generate extended realistic data"""
        # Determine bars based on timeframe
        if timeframe == 'D1':
            bars = years * 365
        elif timeframe == 'H4':
            bars = years * 365 * 6
        else:  # H1
            bars = years * 365 * 24
        
        dates = pd.date_range(end=datetime.now(), periods=bars, freq='D' if timeframe == 'D1' else 'H')
        
        # Enhanced symbol profiles with more realism
        profiles = {
            "US30Z25.sim": {"base": 35000, "vol": 0.015, "trend": 0.0002, "seasonality": 0.001},
            "US100Z25.sim": {"base": 18000, "vol": 0.012, "trend": 0.0003, "seasonality": 0.002},
            "XAUZ25.sim": {"base": 2000, "vol": 0.008, "trend": 0.0001, "seasonality": 0.0005}
        }
        
        profile = profiles.get(symbol, {"base": 10000, "vol": 0.01, "trend": 0.0002, "seasonality": 0.001})
        
        # Generate more realistic price series with seasonality
        prices = [profile["base"]]
        for i in range(1, bars):
            # Add seasonality component
            seasonal = profile["seasonality"] * np.sin(2 * np.pi * i / 252)  # Annual cycles
            change = profile["trend"] + seasonal + np.random.normal(0, profile["vol"])
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, profile["base"] * 0.3))  # Prevent crashes
        
        df = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'tick_volume': [abs(int(np.random.normal(1000, 300))) for _ in prices]
        }, index=dates)
        
        self.log_progress(f"✓ Generated {len(df)} extended bars for {symbol}")
        return df
    
    def calculate_extended_features(self, df, parameters):
        """Calculate comprehensive features for optimization"""
        feature_set = parameters['feature_set']
        
        # Always include core features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Extended moving averages based on parameter set
        lookback_periods = parameters['lookback_periods']
        for period in lookback_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'wma_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=False
            )
        
        # Volatility features
        vol_window = parameters['volatility_window']
        df['volatility'] = df['returns'].rolling(vol_window).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
        
        # RSI with different periods
        rsi_period = parameters['rsi_period']
        df['rsi'] = self.calculate_rsi(df['close'], rsi_period)
        
        # MACD with optimized parameters
        macd_fast = parameters.get('macd_fast', 12)
        macd_slow = parameters.get('macd_slow', 26)
        ema_fast = df['close'].ewm(span=macd_fast).mean()
        ema_slow = df['close'].ewm(span=macd_slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Advanced features for comprehensive sets
        if feature_set in ['advanced', 'comprehensive', 'technical_only']:
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * bb_std
            df['bb_lower'] = df['bb_middle'] - 2 * bb_std
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['stochastic'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            
            # Price channels
            df['channel_high'] = df['high'].rolling(20).max()
            df['channel_low'] = df['low'].rolling(20).min()
            df['channel_position'] = (df['close'] - df['channel_low']) / (df['channel_high'] - df['channel_low'])
        
        if feature_set == 'comprehensive':
            # Additional comprehensive features
            df['momentum'] = df['close'] / df['close'].shift(5) - 1
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['true_range'].rolling(14).mean()
            
            # Volume analysis (simulated)
            if 'tick_volume' in df.columns:
                df['volume_sma'] = df['tick_volume'].rolling(20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
                df['obv'] = (df['tick_volume'] * np.sign(df['close'].diff())).cumsum()
        
        return df.dropna()
    
    def calculate_rsi(self, prices, period):
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def run_parameter_backtest(self, symbol, parameters, years=5):
        """Run backtest with specific parameters"""
        try:
            # Get extended data
            df = self.get_historical_data_extended(symbol, years, parameters['timeframe'])
            if df is None or len(df) < 300:  # Require more data for thorough testing
                return None
            
            # Calculate comprehensive features
            df = self.calculate_extended_features(df, parameters)
            if len(df) < 100:
                return None
            
            # Create target based on horizon
            horizon = parameters['target_horizon']
            df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
            
            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['target', 'time', 'open', 'high', 'low', 'close', 'tick_volume']]
            X = df[feature_cols]
            y = df['target']
            
            # Remove NaN targets
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:  # Require more samples for thorough testing
                return None
            
            # Enhanced train/test split
            train_ratio = parameters['train_ratio']
            split_idx = int(len(X) * train_ratio)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Use more sophisticated model for optimization
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200,  # More trees for better performance
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Enhanced predictions with probability
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            accuracy = (predictions == y_test).mean()
            
            # Enhanced trading simulation
            trading_result = self.simulate_enhanced_trading(df, X_test, predictions, probabilities, split_idx)
            
            return {
                'accuracy': accuracy,
                'total_return': trading_result['total_return'],
                'sharpe_ratio': trading_result['sharpe_ratio'],
                'max_drawdown': trading_result['max_drawdown'],
                'win_rate': trading_result['win_rate'],
                'profit_factor': trading_result['profit_factor'],
                'expectancy': trading_result['expectancy'],
                'training_samples': len(X_train),
                'testing_samples': len(X_test),
                'features_used': len(feature_cols),
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
        except Exception as e:
            self.log_progress(f"Parameter test error for {symbol}: {e}")
            return None
    
    def simulate_enhanced_trading(self, df, X_test, predictions, probabilities, split_idx):
        """Enhanced trading simulation with probability-based decisions"""
        if len(X_test) == 0:
            return {
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 
                'win_rate': 0, 'profit_factor': 0, 'expectancy': 0
            }
        
        test_prices = df['close'].iloc[split_idx:split_idx + len(X_test)]
        initial_balance = 10000
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_balance]
        
        for i, (prediction, probability, price) in enumerate(zip(predictions, probabilities, test_prices)):
            if i == 0:
                continue
            
            # Close previous position if any
            if position != 0:
                pnl = (price - entry_price) / entry_price * (10000 if position > 0 else -10000)
                balance += pnl
                trades.append({
                    'profit': pnl,
                    'win': pnl > 0,
                    'confidence': probability,  # Track confidence of winning trades
                    'duration': i  # Track trade duration
                })
                position = 0
                equity_curve.append(balance)
            
            # Enhanced entry logic using probability thresholds
            if prediction == 1 and probability > 0.6 and balance > 1000:  # Higher threshold
                position = 1
                entry_price = price
                # Position size based on confidence
                position_size = min(1000 * (probability - 0.5) * 2, 2000)  # Scale with confidence
                balance -= position_size
                
            elif prediction == 0 and probability < 0.4 and balance > 1000:  # Short with low probability
                position = -1
                entry_price = price
                position_size = min(1000 * (0.5 - probability) * 2, 2000)
                balance -= position_size
        
        # Close final position
        if position != 0:
            pnl = (test_prices.iloc[-1] - entry_price) / entry_price * (10000 if position > 0 else -10000)
            balance += pnl
            trades.append({
                'profit': pnl,
                'win': pnl > 0,
                'confidence': probabilities[-1] if len(probabilities) > 0 else 0.5
            })
            equity_curve.append(balance)
        
        # Calculate enhanced metrics
        total_return = (balance - initial_balance) / initial_balance
        
        if trades:
            returns = [t['profit'] / 10000 for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
            
            # Max drawdown
            equity_array = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (peak - equity_array) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            win_rate = len([t for t in trades if t['win']]) / len(trades)
            
            # Profit factor (Gross Profit / Gross Loss)
            gross_profit = sum([t['profit'] for t in trades if t['profit'] > 0])
            gross_loss = abs(sum([t['profit'] for t in trades if t['profit'] < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expectancy (Average Win * Win Rate - Average Loss * Loss Rate)
            avg_win = np.mean([t['profit'] for t in trades if t['profit'] > 0]) if any(t['profit'] > 0 for t in trades) else 0
            avg_loss = np.mean([abs(t['profit']) for t in trades if t['profit'] < 0]) if any(t['profit'] < 0 for t in trades) else 0
            loss_rate = 1 - win_rate
            expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
        else:
            sharpe_ratio, max_drawdown, win_rate, profit_factor, expectancy = 0, 0, 0, 0, 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    def optimize_parameters_high_volume(self, symbol, years=5, max_combinations=3000):
        """Run high-volume parameter optimization"""
        self.log_progress(f"🚀 HIGH-VOLUME OPTIMIZATION FOR {symbol}")
        self.log_progress(f"Testing {max_combinations:,} combinations over {years} years")
        
        # Generate parameter combinations
        param_combinations = self.generate_3000_combinations()
        
        best_result = None
        best_params = None
        tested_combinations = 0
        successful_tests = 0
        
        # Test each parameter combination
        for i, params in enumerate(param_combinations):
            tested_combinations += 1
            
            if i % 100 == 0:  # Progress update every 100 tests
                self.log_progress(f"Progress: {i+1}/{len(param_combinations)} combinations tested")
            
            result = self.run_parameter_backtest(symbol, params, years)
            
            if result:
                successful_tests += 1
                
                # Enhanced scoring system
                score = (result['accuracy'] * 0.3 + 
                        result['total_return'] * 0.25 + 
                        result['sharpe_ratio'] * 0.2 + 
                        result['profit_factor'] * 0.15 - 
                        result['max_drawdown'] * 0.1)
                
                result['optimization_score'] = score
                
                # Update best result
                if best_result is None or score > best_result['optimization_score']:
                    best_result = result
                    best_params = params
                    
                    self.log_progress(f"🎯 NEW BEST: Acc={result['accuracy']:.3f}, "
                                   f"Ret={result['total_return']:.2%}, Sharpe={result['sharpe_ratio']:.3f}")
            
        self.log_progress(f"✓ Optimization completed: {successful_tests}/{tested_combinations} successful")
        
        if best_result:
            self.log_progress("\\n🏆 BEST PARAMETERS FOUND:")
            for param, value in best_params.items():
                self.log_progress(f"  {param}: {value}")
            
            self.log_progress("\\n📊 BEST PERFORMANCE:")
            self.log_progress(f"  Accuracy: {best_result['accuracy']:.3f}")
            self.log_progress(f"  Return: {best_result['total_return']:.2%}")
            self.log_progress(f"  Sharpe: {best_result['sharpe_ratio']:.3f}")
            self.log_progress(f"  Profit Factor: {best_result['profit_factor']:.2f}")
            self.log_progress(f"  Win Rate: {best_result['win_rate']:.1%}")
            
            return best_params, best_result
        else:
            self.log_progress("❌ No successful parameter combinations found")
            return None, None
    
    def run_comprehensive_high_volume_optimization(self, years=5, save_best_models=True):
        """Run comprehensive optimization with 3000 combinations per symbol"""
        self.log_progress("🚀 STARTING HIGH-VOLUME PARAMETER OPTIMIZATION")
        self.log_progress("=" * 60)
        self.log_progress(f"Testing 3000+ combinations per symbol over {years} years")
        
        # Try MT5 connection
        try:
            mt5.initialize()
            mt5.login(1600038177, server="OANDA-Demo-1")
            mt5_connected = True
        except:
            mt5_connected = False
        
        self.optimization_results = {}
        
        # Optimize each symbol with high volume
        for symbol in self.symbols:
            self.log_progress(f"\\n🎯 HIGH-VOLUME OPTIMIZATION FOR {symbol}")
            
            best_params, best_result = self.optimize_parameters_high_volume(symbol, years)
            
            if best_params and best_result:
                self.optimization_results[symbol] = {
                    'parameters': best_params,
                    'performance': best_result
                }
                
                # Save best parameters
                param_path = os.path.join(self.model_save_path, f"best_params_{symbol}.pkl")
                with open(param_path, 'wb') as f:
                    pickle.dump(best_params, f)
                
                self.log_progress(f"✓ Best parameters saved for {symbol}")
        
        # Generate optimization report
        self.generate_optimization_report()
        
        if mt5_connected:
            try:
                mt5.shutdown()
            except:
                pass
        
        successful_optimizations = len(self.optimization_results)
        self.log_progress(f"✓ High-volume optimization completed: "
                         f"{successful_optimizations}/{len(self.symbols)} symbols optimized")
        
        return successful_optimizations > 0
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        if not self.optimization_results:
            self.log_progress("❌ No optimization results to report")
            return
        
        self.log_progress("\\n" + "=" * 60)
        self.log_progress("📊 HIGH-VOLUME OPTIMIZATION REPORT")
        self.log_progress("=" * 60)
        
        for symbol, results in self.optimization_results.items():
            self.log_progress(f"\\n{symbol}:")
            perf = results['performance']
            self.log_progress(f"  Accuracy: {perf['accuracy']:.3f}")
            self.log_progress(f"  Return: {perf['total_return']:.2%}")
            self.log_progress(f"  Sharpe: {perf['sharpe_ratio']:.3f}")
            self.log_progress(f"  Profit Factor: {perf['profit_factor']:.2f}")
            self.log_progress(f"  Optimization Score: {perf['optimization_score']:.3f}")
            
            params = results['parameters']
            self.log_progress("  Best Parameters:")
            for param, value in params.items():
                self.log_progress(f"    {param}: {value}")
        
        # Save detailed report
        try:
            with open("high_volume_optimization_report.txt", 'w') as f:
                f.write("FTMO AI High-Volume Parameter Optimization Report\\n")
                f.write("=" * 60 + "\\n")
                for symbol, results in self.optimization_results.items():
                    perf = results['performance']
                    f.write(f"\\n{symbol}:\\n")
                    f.write(f"  Accuracy: {perf['accuracy']:.3f}\\n")
                    f.write(f"  Return: {perf['total_return']:.2%}\\n")
                    f.write(f"  Sharpe: {perf['sharpe_ratio']:.3f}\\n")
                    f.write(f"  Profit Factor: {perf['profit_factor']:.2f}\\n")
                    f.write(f"  Score: {perf['optimization_score']:.3f}\\n")
            
            self.log_progress("✓ Detailed report saved to high_volume_optimization_report.txt")
        except Exception as e:
            self.log_progress(f"⚠️ Error saving report: {e}")

def run_high_volume_optimization():
    """Main function for high-volume optimization"""
    print("🎯 HIGH-VOLUME PARAMETER OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    optimizer = HighVolumeOptimizer()
    
    def progress_callback(message):
        print(f"▶ {message}")
    
    optimizer.set_progress_callback(progress_callback)
    
    # Run optimization with 3 years for reasonable time
    success = optimizer.run_comprehensive_high_volume_optimization(years=3, save_best_models=True)
    
    if success:
        print("\\n🎉 HIGH-VOLUME OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("Best parameters saved for live trading")
    else:
        print("\\n⚠️ Optimization completed with limited results")

if __name__ == "__main__":
    run_high_volume_optimization()
