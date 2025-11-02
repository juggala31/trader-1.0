import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveSymbolAnalyzer:
    def __init__(self):
        # OANDA-specific symbols for MT5
        self.precious_metals = [
            'XAUUSD', 'XAGUSD',  # Gold, Silver
            'XPTUSD', 'XPDUSD',  # Platinum, Palladium
            'XAUAUD', 'XAGEUR',  # Cross pairs
        ]
        
        self.indices = [
            # US Indices
            'US30', 'US500', 'US100', 'US2000', 'USTEC', 'USDOLLAR',
            # European Indices
            'DE30', 'UK100', 'FR40', 'EU50', 'IT40', 'ES35',
            # Asian Indices
            'JP225', 'HK50', 'AUS200', 'SING50',
            # Other
            'CHINA50', 'BRENT', 'WTI'
        ]
        
        self.timeframes = {
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        self.results = {}
        
    def connect_mt5_oanda(self):
        """Connect to MT5 with OANDA symbols"""
        try:
            if not mt5.initialize():
                print("❌ MT5 initialization failed")
                return False
            
            # Get all available symbols
            all_symbols = mt5.symbols_get()
            available_symbols = [s.name for s in all_symbols]
            
            print("✅ MT5 connected successfully")
            print(f"📊 Total symbols available: {len(available_symbols)}")
            
            # Filter to find our target symbols
            self.available_precious_metals = [s for s in self.precious_metals if s in available_symbols]
            self.available_indices = [s for s in self.indices if s in available_symbols]
            
            print(f"💎 Available Precious Metals: {len(self.available_precious_metals)}")
            for pm in self.available_precious_metals:
                print(f"   ✅ {pm}")
            
            print(f"📈 Available Indices: {len(self.available_indices)}")
            for idx in self.available_indices:
                print(f"   ✅ {idx}")
            
            if not self.available_precious_metals and not self.available_indices:
                print("❌ No target symbols found")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_2_year_data(self, symbol, timeframe):
        """Get 2 years of historical data"""
        try:
            print(f"📥 Fetching 2 years of {symbol} on {timeframe}...")
            
            # Calculate start date (2 years ago)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years
            
            # Estimate required bars (conservative estimate)
            if timeframe == mt5.TIMEFRAME_H1:
                bars_needed = 365 * 24 * 2  # ~17,520 bars
            elif timeframe == mt5.TIMEFRAME_H4:
                bars_needed = 365 * 6 * 2   # ~4,380 bars
            else:  # D1
                bars_needed = 365 * 2       # ~730 bars
            
            # Try to get data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, min(bars_needed, 20000))
            if rates is None:
                # Alternative method: copy by range
                rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                print(f"   ❌ No data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            if len(df) < 100:  # Minimum data requirement
                print(f"   ❌ Insufficient data: {len(df)} bars")
                return None
            
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Calculate actual date range
            actual_start = df.index.min()
            actual_end = df.index.max()
            days_covered = (actual_end - actual_start).days
            
            print(f"   ✅ Loaded {len(df):,} bars covering {days_covered} days")
            print(f"   📅 Date range: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Error fetching {symbol}: {e}")
            return None
    
    def calculate_performance_metrics(self, df, symbol, timeframe_name):
        """Calculate comprehensive performance metrics"""
        if df is None or len(df) < 200:
            return None
        
        try:
            # Basic price calculations
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility metrics
            df['volatility_20'] = df['returns'].rolling(20).std()
            df['volatility_50'] = df['returns'].rolling(50).std()
            
            # Trend indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
            
            # Support/Resistance
            df['resistance_50'] = df['high'].rolling(50).max()
            df['support_50'] = df['low'].rolling(50).min()
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 100:
                return None
            
            # Calculate comprehensive metrics
            metrics = {}
            
            # Basic statistics
            metrics['total_bars'] = len(df)
            metrics['date_range_days'] = (df.index.max() - df.index.min()).days
            metrics['avg_daily_bars'] = len(df) / metrics['date_range_days'] if metrics['date_range_days'] > 0 else 0
            
            # Return metrics
            total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            metrics['total_return_pct'] = total_return
            
            # Annualized return
            years = metrics['date_range_days'] / 365.25
            metrics['annualized_return_pct'] = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
            
            # Volatility metrics
            metrics['avg_daily_volatility'] = df['volatility_20'].mean() * 100  # As percentage
            metrics['max_daily_volatility'] = df['volatility_20'].max() * 100
            metrics['volatility_stability'] = df['volatility_20'].std() / df['volatility_20'].mean() if df['volatility_20'].mean() > 0 else 0
            
            # Trend metrics
            metrics['trend_strength'] = (df['sma_20'].iloc[-1] - df['sma_200'].iloc[-1]) / df['sma_200'].iloc[-1] * 100
            metrics['avg_trend_strength'] = ((df['sma_20'] - df['sma_200']) / df['sma_200']).mean() * 100
            
            # Range metrics (for mean reversion strategies)
            current_price = df['close'].iloc[-1]
            resistance = df['resistance_50'].iloc[-1]
            support = df['support_50'].iloc[-1]
            metrics['price_in_range'] = (current_price - support) / (resistance - support) if resistance > support else 0.5
            
            # Consistency metrics
            positive_returns = (df['returns'] > 0).sum()
            metrics['positive_return_ratio'] = positive_returns / len(df)
            
            # Drawdown analysis
            df['peak'] = df['close'].expanding().max()
            df['drawdown'] = (df['close'] - df['peak']) / df['peak'] * 100
            metrics['max_drawdown_pct'] = df['drawdown'].min()
            metrics['avg_drawdown_pct'] = df['drawdown'].mean()
            
            # Sharpe ratio (simplified)
            avg_return = df['log_returns'].mean() * 252  # Annualized
            std_return = df['log_returns'].std() * np.sqrt(252)  # Annualized
            metrics['sharpe_ratio'] = avg_return / std_return if std_return > 0 else 0
            
            # Market regime detection
            bull_market = (df['sma_20'] > df['sma_50']).sum() / len(df)
            metrics['bull_market_ratio'] = bull_market
            
            print(f"   📊 {symbol} on {timeframe_name}: Return={metrics['total_return_pct']:.2f}%, "
                  f"Volatility={metrics['avg_daily_volatility']:.2f}%, Sharpe={metrics['sharpe_ratio']:.2f}")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Metrics calculation error for {symbol}: {e}")
            return None
    
    def score_symbol_performance(self, metrics):
        """Score symbol based on multiple performance factors"""
        if metrics is None:
            return 0
        
        try:
            # Weighted scoring system
            score = 0
            
            # Return-based scoring (40% weight)
            return_score = min(metrics['annualized_return_pct'] * 2, 100)  # Cap at 100
            score += return_score * 0.4
            
            # Risk-adjusted scoring (30% weight)
            sharpe_score = min(metrics['sharpe_ratio'] * 20, 100)  # Good Sharpe > 5
            score += sharpe_score * 0.3
            
            # Stability scoring (20% weight)
            volatility_score = 100 - min(metrics['avg_daily_volatility'] * 10, 100)  # Lower volatility better
            stability_score = 100 - (metrics['volatility_stability'] * 100)
            stability_combined = (volatility_score + stability_score) / 2
            score += stability_combined * 0.2
            
            # Trend scoring (10% weight)
            trend_score = min(abs(metrics['trend_strength']) * 10, 100)
            score += trend_score * 0.1
            
            return max(0, min(score, 100))  # Ensure score between 0-100
            
        except:
            return 0
    
    def analyze_symbol_category(self, symbols, category_name):
        """Analyze all symbols in a category"""
        print(f"\n{'='*80}")
        print(f"🔍 ANALYZING {category_name.upper()}")
        print(f"{'='*80}")
        
        category_results = {}
        
        for symbol in symbols:
            print(f"\n🎯 Analyzing {symbol}")
            print("-" * 40)
            
            symbol_results = {}
            
            for tf_name, tf_value in self.timeframes.items():
                # Get 2 years of data
                df = self.get_2_year_data(symbol, tf_value)
                
                if df is None:
                    symbol_results[tf_name] = {
                        'status': 'No data',
                        'metrics': None,
                        'score': 0
                    }
                    continue
                
                # Calculate performance metrics
                metrics = self.calculate_performance_metrics(df, symbol, tf_name)
                
                if metrics is None:
                    symbol_results[tf_name] = {
                        'status': 'Metrics failed',
                        'metrics': None,
                        'score': 0
                    }
                    continue
                
                # Score the symbol
                score = self.score_symbol_performance(metrics)
                
                symbol_results[tf_name] = {
                    'status': 'Success',
                    'metrics': metrics,
                    'score': score
                }
            
            category_results[symbol] = symbol_results
        
        return category_results
    
    def run_comprehensive_analysis(self):
        """Run analysis on all symbol categories"""
        if not self.connect_mt5_oanda():
            return False
        
        print("🚀 COMPREHENSIVE SYMBOL ANALYSIS")
        print("📅 2-YEAR BACKTESTING PERIOD")
        print("💎 Precious Metals + 📈 Indices")
        print("=" * 80)
        
        # Analyze precious metals
        if self.available_precious_metals:
            pm_results = self.analyze_symbol_category(self.available_precious_metals, "Precious Metals")
            self.results['precious_metals'] = pm_results
        else:
            print("❌ No precious metals symbols available")
            self.results['precious_metals'] = {}
        
        # Analyze indices
        if self.available_indices:
            idx_results = self.analyze_symbol_category(self.available_indices, "Indices")
            self.results['indices'] = idx_results
        else:
            print("❌ No indices symbols available")
            self.results['indices'] = {}
        
        self.generate_comprehensive_report()
        return True
    
    def generate_comprehensive_report(self):
        """Generate detailed analysis report"""
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE SYMBOL ANALYSIS REPORT")
        print("=" * 100)
        print("2-YEAR PERFORMANCE ANALYSIS FOR AI TRADING")
        print("=" * 100)
        
        all_symbols_data = []
        
        # Combine all symbols with their scores
        for category, symbols_data in self.results.items():
            for symbol, timeframes_data in symbols_data.items():
                for tf_name, result in timeframes_data.items():
                    if result['status'] == 'Success' and result['metrics']:
                        metrics = result['metrics']
                        all_symbols_data.append({
                            'symbol': symbol,
                            'timeframe': tf_name,
                            'category': category,
                            'score': result['score'],
                            'annual_return': metrics['annualized_return_pct'],
                            'volatility': metrics['avg_daily_volatility'],
                            'sharpe': metrics['sharpe_ratio'],
                            'max_drawdown': metrics['max_drawdown_pct'],
                            'trend_strength': metrics['trend_strength'],
                            'data_points': metrics['total_bars']
                        })
        
        if not all_symbols_data:
            print("❌ No successful analyses to report")
            return
        
        # Create DataFrame for sorting
        df_report = pd.DataFrame(all_symbols_data)
        
        # Top performers by score
        print("\n🏆 TOP 10 SYMBOLS FOR AI TRADING (Overall Score)")
        print("-" * 80)
        top_symbols = df_report.sort_values('score', ascending=False).head(10)
        
        for idx, row in top_symbols.iterrows():
            print(f"{idx+1:2d}. {row['symbol']:8} ({row['timeframe']}) | "
                  f"Score: {row['score']:5.1f} | "
                  f"Return: {row['annual_return']:6.2f}% | "
                  f"Vol: {row['volatility']:5.2f}% | "
                  f"Sharpe: {row['sharpe']:5.2f}")
        
        # Best by category
        print("\n💎 BEST PRECIOUS METALS")
        print("-" * 40)
        pm_best = df_report[df_report['category'] == 'precious_metals'].sort_values('score', ascending=False).head(5)
        for idx, row in pm_best.iterrows():
            print(f"{row['symbol']} ({row['timeframe']}): Score={row['score']:.1f}, Return={row['annual_return']:.2f}%")
        
        print("\n📈 BEST INDICES")
        print("-" * 40)
        idx_best = df_report[df_report['category'] == 'indices'].sort_values('score', ascending=False).head(5)
        for idx, row in idx_best.iterrows():
            print(f"{row['symbol']} ({row['timeframe']}): Score={row['score']:.1f}, Return={row['annual_return']:.2f}%")
        
        # Risk-adjusted recommendations
        print("\n🎯 RISK-ADJUSTED RECOMMENDATIONS (High Sharpe Ratio)")
        print("-" * 60)
        sharpe_best = df_report[df_report['sharpe'] > 0].sort_values('sharpe', ascending=False).head(5)
        for idx, row in sharpe_best.iterrows():
            print(f"{row['symbol']} ({row['timeframe']}): Sharpe={row['sharpe']:.2f}, Return={row['annual_return']:.2f}%")
        
        # Low volatility recommendations
        print("\n🛡️ LOW VOLATILITY RECOMMENDATIONS")
        print("-" * 50)
        low_vol = df_report.sort_values('volatility').head(5)
        for idx, row in low_vol.iterrows():
            print(f"{row['symbol']} ({row['timeframe']}): Vol={row['volatility']:.2f}%, Return={row['annual_return']:.2f}%")
        
        self.save_detailed_analysis()
    
    def save_detailed_analysis(self):
        """Save comprehensive analysis to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"symbol_analysis_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("COMPREHENSIVE SYMBOL ANALYSIS REPORT\\n")
            f.write("=" * 70 + "\\n")
            f.write(f"Analysis Date: {datetime.now()}\\n")
            f.write("Period: 2 Years\\n")
            f.write("Broker: OANDA MT5\\n\\n")
            
            for category, symbols_data in self.results.items():
                f.write(f"{category.upper()}\\n")
                f.write("-" * 50 + "\\n")
                
                for symbol, timeframes_data in symbols_data.items():
                    f.write(f"{symbol}:\\n")
                    for tf_name, result in timeframes_data.items():
                        f.write(f"  {tf_name}: {result['status']}\\n")
                        if result['status'] == 'Success':
                            metrics = result['metrics']
                            f.write(f"    Score: {result['score']:.1f}\\n")
                            f.write(f"    Annual Return: {metrics['annualized_return_pct']:.2f}%\\n")
                            f.write(f"    Volatility: {metrics['avg_daily_volatility']:.2f}%\\n")
                            f.write(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\\n")
                            f.write(f"    Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\\n")
                            f.write(f"    Data Points: {metrics['total_bars']:,}\\n")
                    f.write("\\n")
            
            f.write("RECOMMENDATION: Focus on symbols with high scores and good risk-adjusted returns\\n")
        
        print(f"\\n💾 Full analysis saved to: {filename}")

# Execute comprehensive analysis
if __name__ == "__main__":
    print("🚀 Starting Comprehensive Symbol Analysis...")
    print("This will analyze ALL precious metals and indices with 2 years of data")
    print("Perfect for determining the best symbols for your AI trading system\\n")
    
    analyzer = ComprehensiveSymbolAnalyzer()
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("\\n✅ Symbol analysis completed successfully!")
        print("Check the report for data-driven symbol recommendations")
    else:
        print("\\n❌ Analysis failed")
