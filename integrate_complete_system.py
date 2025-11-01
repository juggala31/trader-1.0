# integrate_complete_system.py (Updated)
import sys
import time
from regime_ensemble_integration import RegimeEnhancedEnsemble, integrate_regime_risk_management
from advanced_risk_management import AdvancedRiskManager
from enhanced_mt5_integration import EnhancedMT5Integration
from live_performance_optimizer import LivePerformanceOptimizer

class CompleteTradingSystem:
    def __init__(self):
        print(\"Initializing Complete FTMO Trading System with Market Regime Detection...\")
        
        # Initialize components
        self.mt5_integration = EnhancedMT5Integration()
        self.regime_ensemble = RegimeEnhancedEnsemble()
        self.risk_manager = integrate_regime_risk_management()
        self.performance_optimizer = LivePerformanceOptimizer()
        
        # System status
        self.system_status = {
            'regime_detection_initialized': False,
            'last_regime_analysis': None,
            'active_trades': [],
            'performance_metrics': {}
        }
    
    def initialize_system(self):
        \"\"\"Initialize all system components\"\"\"
        try:
            # Initialize MT5 connection
            if not self.mt5_integration.initialize_connection():
                print(\"Failed to initialize MT5 connection\")
                return False
            
            # Initialize regime detection
            print(\"Initializing market regime detection...\")
            self.regime_ensemble.initialize_regime_detection()
            self.system_status['regime_detection_initialized'] = True
            
            # Initialize risk management
            self.risk_manager.initialize()
            
            print(\"System initialization completed successfully!\")
            return True
            
        except Exception as e:
            print(f\"System initialization failed: {e}\")
            return False
    
    def run_trading_cycle(self):
        \"\"\"Execute one complete trading cycle\"\"\"
        try:
            # Get current market regime
            regime_analysis = self.regime_ensemble.get_current_regime_analysis()
            self.system_status['last_regime_analysis'] = regime_analysis
            
            if regime_analysis:
                print(f\"Current Market Regime: {regime_analysis['regime_label']} \" 
                      f\"(Confidence: {regime_analysis['confidence']:.2%})\")
            
            # Generate regime-enhanced signals
            signals = self.regime_ensemble.generate_regime_enhanced_signals()
            
            if signals:
                print(f\"Generated {len(signals)} trading signals\")
                
                # Execute trades with regime-aware risk management
                for signal in signals:
                    self.execute_trade_with_regime_awareness(signal, regime_analysis)
            
            # Update performance metrics
            self.update_performance_metrics()
            
            return True
            
        except Exception as e:
            print(f\"Trading cycle error: {e}\")
            return False
    
    def execute_trade_with_regime_awareness(self, signal, regime_analysis):
        \"\"\"Execute trade with regime-aware parameters\"\"\"
        try:
            # Get regime-specific parameters
            regime_params = self.regime_ensemble.regime_trader.get_regime_specific_parameters(
                regime_analysis if regime_analysis else {'regime': 1, 'confidence': 0.5}
            )
            
            # Adjust signal based on regime
            symbol = signal['symbol']
            direction = signal['direction']
            strength = signal['strength']
            
            # Calculate regime-aware position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, strength, 
                regime_analysis['regime'] if regime_analysis else 1
            )
            
            # Calculate regime-aware stop loss
            current_price = self.mt5_integration.get_current_price(symbol)
            stop_loss = self.risk_manager.get_stop_loss_level(
                symbol, current_price, direction,
                regime_analysis['regime'] if regime_analysis else 1
            )
            
            # Execute trade
            trade_result = self.mt5_integration.execute_trade(
                symbol=symbol,
                order_type=direction,
                volume=position_size,
                stop_loss=stop_loss,
                take_profit=current_price * (1 + 0.02) if direction == 'BUY' else current_price * (1 - 0.02)
            )
            
            if trade_result:
                self.system_status['active_trades'].append({
                    'ticket': trade_result.order,
                    'symbol': symbol,
                    'regime': regime_analysis['regime_label'] if regime_analysis else 'Unknown',
                    'timestamp': time.time()
                })
                print(f\"Trade executed successfully in {regime_analysis['regime_label'] if regime_analysis else 'Unknown'} regime\")
            
        except Exception as e:
            print(f\"Trade execution error: {e}\")
    
    def update_performance_metrics(self):
        \"\"\"Update system performance metrics\"\"\"
        try:
            # Get regime performance metrics
            regime_metrics = self.regime_ensemble.get_regime_performance_metrics()
            
            # Get overall performance metrics
            overall_metrics = self.performance_optimizer.get_performance_metrics()
            
            self.system_status['performance_metrics'] = {
                'regime_metrics': regime_metrics,
                'overall_metrics': overall_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f\"Performance metrics update error: {e}\")
    
    def get_system_status(self):
        \"\"\"Get current system status\"\"\"
        return self.system_status
    
    def run_continuous_trading(self, interval_minutes=5):
        \"\"\"Run continuous trading with specified interval\"\"\"
        print(f\"Starting continuous trading with {interval_minutes}-minute intervals...\")
        
        while True:
            try:
                cycle_start = time.time()
                
                # Run trading cycle
                self.run_trading_cycle()
                
                # Calculate time until next cycle
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, interval_minutes * 60 - cycle_duration)
                
                print(f\"Cycle completed. Sleeping for {sleep_time:.1f} seconds...\")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print(\"Trading stopped by user\")
                break
            except Exception as e:
                print(f\"Continuous trading error: {e}\")
                time.sleep(60)  # Wait 1 minute before retrying

# Main execution
if __name__ == \"__main__\":
    trading_system = CompleteTradingSystem()
    
    if trading_system.initialize_system():
        # Run one cycle immediately
        trading_system.run_trading_cycle()
        
        # Start continuous trading
        trading_system.run_continuous_trading(interval_minutes=10)
    else:
        print(\"System initialization failed. Please check MT5 connection and configuration.\")
