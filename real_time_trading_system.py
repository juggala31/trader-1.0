import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from real_time_mt5_integration import RealTimeMT5Integration, RealTimeRiskManager
from market_regime_detector import MarketRegimeDetector
from reinforcement_learning import RLEnhancedTrading

class RealTimeTradingSystem:
    def __init__(self, symbols=None):
        self.symbols = symbols or ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]
        self.mt5 = RealTimeMT5Integration(login=1600038177, server="OANDA-Demo-1")
        self.risk_manager = RealTimeRiskManager(self.mt5)
        self.regime_detector = MarketRegimeDetector()
        self.rl_trader = RLEnhancedTrading(self.regime_detector)
        self.performance_log = []
        
    def initialize_system(self):
        """Initialize system with real MT5 data"""
        print("Initializing Real-Time Trading System...")
        
        # Connect to MT5
        if not self.mt5.initialize_connection():
            print("Using simulated mode (MT5 connection failed)")
            # Fallback to simulated data
            return self._initialize_simulated()
        
        # Initialize risk management
        self.risk_manager.initialize_daily_tracking()
        
        # Initialize AI components
        historical_data = self._get_real_historical_data()
        if historical_data is not None:
            self.regime_detector.train_model(historical_data)
            print("✓ Market regime detection trained on real data")
        else:
            # Fallback to simulated data
            historical_data = self._generate_sample_data(1000)
            self.regime_detector.train_model(historical_data)
            print("✓ Market regime detection trained on simulated data")
        
        # Show initial account status
        balance = self.mt5.get_current_balance()
        equity = self.mt5.get_current_equity()
        print(f"💰 Initial Balance: ${balance:.2f}")
        print(f"📊 Initial Equity: ${equity:.2f}")
        print("✓ Real-time trading system initialized")
        
        return True
    
    def _initialize_simulated(self):
        """Initialize with simulated data when MT5 is not available"""
        historical_data = self._generate_sample_data(1000)
        self.regime_detector.train_model(historical_data)
        print("✓ System initialized in simulated mode")
        return True
    
    def _get_real_historical_data(self, periods=1000):
        """Get real historical data from MT5"""
        try:
            if not self.mt5.connected:
                return None
            
            # Get data for the first symbol as example
            symbol = self.symbols[0]
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, periods)
            
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df['close']
            
        except Exception as e:
            print(f"Error getting real historical data: {e}")
            return None
    
    def _generate_sample_data(self, periods):
        """Generate sample data for fallback"""
        dates = pd.date_range(start="2020-01-01", periods=periods, freq="D")
        prices = 100 + np.cumsum(np.random.randn(periods) * 0.3 + 0.001)
        return pd.Series(prices, index=dates)
    
    def generate_trading_signals(self):
        """Generate signals using real account data"""
        signals = []
        
        # Check risk limits before generating signals
        if not (self.risk_manager.check_daily_loss_limit() and 
                self.risk_manager.check_total_drawdown_limit()):
            print("🚨 Trading halted due to risk limits")
            return []
        
        for symbol in self.symbols:
            market_data = self._get_current_market_data(symbol)
            
            # Generate RL signal
            signal, state, action = self.rl_trader.generate_rl_signals(
                market_data, self._get_current_position(symbol)
            )
            
            # Calculate position size based on real balance
            position_size = self.risk_manager.calculate_safe_position_size(
                symbol, signal.get('strength', 0.5)
            )
            
            signal.update({
                'symbol': symbol,
                'position_size': position_size,
                'rl_state': state,
                'rl_action': action
            })
            
            signals.append(signal)
        
        return signals
    
    def _get_current_market_data(self, symbol):
        """Get current market data"""
        # Try to get real data first
        if self.mt5.connected:
            try:
                # Get current price
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    # Create price series for regime detection (last 50 prices)
                    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 50)
                    if rates:
                        prices = [rate[4] for rate in rates]  # Close prices
                        price_series = pd.Series(prices)
                        
                        regime_pred = self.regime_detector.predict_regime(price_series)
                        
                        returns = price_series.pct_change().dropna()
                        trend = (price_series.iloc[-1] / price_series.iloc[0] - 1) if len(price_series) > 1 else 0
                        volatility = returns.std() if len(returns) > 1 else 0.01
                        
                        return {
                            'price_data': price_series,
                            'price': tick.last,
                            'regime': regime_pred['regime'],
                            'trend_strength': abs(trend),
                            'volatility_level': min(1.0, volatility * 10),
                            'rsi': 50,  # Simplified for demo
                            'momentum': trend,
                            'symbol': symbol
                        }
            except Exception as e:
                print(f"Error getting real market data: {e}")
        
        # Fallback to simulated data
        price_data = self._generate_sample_data(50)
        regime_pred = self.regime_detector.predict_regime(price_data)
        
        returns = price_data.pct_change().dropna()
        trend = (price_data.iloc[-1] / price_data.iloc[0] - 1) if len(price_data) > 1 else 0
        volatility = returns.std() if len(returns) > 1 else 0.01
        
        return {
            'price_data': price_data,
            'price': price_data.iloc[-1],
            'regime': regime_pred['regime'],
            'trend_strength': abs(trend),
            'volatility_level': min(1.0, volatility * 10),
            'rsi': 50,
            'momentum': trend,
            'symbol': symbol
        }
    
    def _get_current_position(self, symbol):
        """Get current position for symbol"""
        if self.mt5.connected:
            positions = self.mt5.get_open_positions()
            if not positions.empty:
                symbol_positions = positions[positions['symbol'] == symbol]
                if not symbol_positions.empty:
                    # Return net position (1 for long, -1 for short, 0 for flat)
                    return 1 if symbol_positions['type'].iloc[0] == 0 else -1
        return 0
    
    def execute_trade(self, signal):
        """Execute trade with real MT5 connection"""
        if signal['position_size'] <= 0 or signal['action'] == 'HOLD':
            print(f"No trade for {signal['symbol']} (zero position size or HOLD action)")
            return {'profit': 0, 'duration': 0, 'risk': 0}
        
        if self.mt5.connected:
            # Real execution - convert dollars to lots (simplified)
            # In real trading, you'd need proper lot size calculation
            volume = max(0.01, signal['position_size'] / 10000)  # Simplified conversion
            
            result = self.mt5.execute_trade(
                symbol=signal['symbol'],
                order_type=signal['action'],
                volume=volume,
                stop_loss=0,  # Calculate proper SL
                take_profit=0  # Calculate proper TP
            )
            
            if result:
                trade_result = {
                    'profit': result.profit,
                    'duration': 1,
                    'risk': signal['position_size'] * 0.01
                }
                print(f"Real trade executed for {signal['symbol']}")
            else:
                print(f"Real trade failed for {signal['symbol']}, using simulation")
                trade_result = self._simulate_trade(signal)
        else:
            # Simulated execution
            trade_result = self._simulate_trade(signal)
        
        return trade_result
    
    def _simulate_trade(self, signal):
        """Simulate trade for demo purposes"""
        base_profit = np.random.normal(0.001, 0.006)
        multiplier = signal.get('size_multiplier', 1.0)
        
        if signal['action'] == 'BUY':
            profit = base_profit * multiplier
        elif signal['action'] == 'SELL':
            profit = -base_profit * multiplier
        else:
            profit = 0
        
        return {
            'profit': profit * signal['position_size'],
            'duration': max(1, int(np.random.exponential(1.5))),
            'risk': signal['position_size'] * 0.01
        }
    
    def run_trading_cycle(self):
        """Run one complete trading cycle with real data"""
        print("\\n=== REAL-TIME TRADING CYCLE ===")
        
        # Update account info
        balance = self.mt5.get_current_balance()
        equity = self.mt5.get_current_equity()
        profit = self.mt5.get_current_profit()
        
        print(f"Account Balance: ${balance:.2f}")
        print(f"Account Equity: ${equity:.2f}")
        print(f"Floating P&L: ${profit:.2f}")
        
        # Generate and execute signals
        signals = self.generate_trading_signals()
        
        total_cycle_profit = 0
        for signal in signals:
            print(f"\\n{signal['symbol']}:")
            print(f"  RL Action: {signal['rl_action']}")
            print(f"  Market Regime: {signal.get('market_regime', 'Unknown')}")
            print(f"  Position Size: ${signal['position_size']:.2f}")
            
            trade_result = self.execute_trade(signal)
            total_cycle_profit += trade_result['profit']
            
            print(f"  Trade Result: ${trade_result['profit']:.2f}")
        
        # Log performance
        self.performance_log.append({
            'timestamp': pd.Timestamp.now(),
            'cycle_profit': total_cycle_profit,
            'balance': balance,
            'equity': equity,
            'signals_generated': len(signals)
        })
        
        print(f"\\nCycle Profit: ${total_cycle_profit:.2f}")
        
        # Risk report
        risk_report = self.risk_manager.get_risk_report()
        print(f"Risk Ratio: {risk_report['risk_ratio']:.2%}")
        print(f"Daily P&L: ${risk_report['daily_pnl']:.2f}")
    
    def get_system_metrics(self):
        """Get comprehensive system metrics"""
        rl_metrics = self.rl_trader.get_learning_metrics()
        risk_report = self.risk_manager.get_risk_report()
        portfolio = self.mt5.get_portfolio_summary()
        
        return {
            'account': {
                'balance': risk_report['current_balance'],
                'equity': risk_report['current_equity'],
                'floating_pnl': risk_report['floating_pnl'],
                'daily_pnl': risk_report['daily_pnl']
            },
            'risk_management': risk_report,
            'rl_learning': rl_metrics,
            'portfolio': portfolio,
            'performance_history': len(self.performance_log)
        }
    
    def shutdown(self):
        """Shutdown the system"""
        self.mt5.shutdown()
        print("Trading system shutdown")

def demo_real_time_system():
    """Demo the real-time trading system"""
    print("🚀 REAL-TIME FTMO TRADING SYSTEM DEMO")
    print("=" * 60)
    
    system = RealTimeTradingSystem()
    
    if system.initialize_system():
        # Run multiple trading cycles
        for cycle in range(3):
            print(f"\\n--- Trading Cycle {cycle + 1} ---")
            system.run_trading_cycle()
        
        # Final metrics
        metrics = system.get_system_metrics()
        print("\\n" + "=" * 60)
        print("🎯 REAL-TIME SYSTEM METRICS:")
        print("=" * 60)
        print(f"Final Balance: ${metrics['account']['balance']:.2f}")
        print(f"Final Equity: ${metrics['account']['equity']:.2f}")
        print(f"Total Daily P&L: ${metrics['account']['daily_pnl']:.2f}")
        print(f"RL Success Rate: {metrics['rl_learning']['success_rate']:.2%}")
        print(f"Risk Ratio: {metrics['risk_management']['risk_ratio']:.2%}")
        
        system.shutdown()
    else:
        print("System initialization failed")

if __name__ == "__main__":
    demo_real_time_system()
