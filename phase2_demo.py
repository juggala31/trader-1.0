# FTMO Phase 2 Demo Trading - Simulated Environment
import time
import random
from datetime import datetime
from ftmo_phase2_system import FTMO_Phase2_System
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DemoTrader:
    def __init__(self):
        self.phase2_system = FTMO_Phase2_System("1600038177", "200k")
        self.demo_mode = True
        self.trade_count = 0
        
    def simulate_market_data(self):
        """Simulate market data for demo"""
        # Simulate price movements
        symbols = ["US30", "US100", "UAX"]
        simulated_data = {}
        
        for symbol in symbols:
            # Simulate price between 35000 and 38000 for US30, etc.
            base_price = 35000 if symbol == "US30" else 15000 if symbol == "US100" else 8000
            price_variation = random.uniform(-0.02, 0.02)  # ±2% variation
            current_price = base_price * (1 + price_variation)
            
            simulated_data[symbol] = {
                'price': current_price,
                'trend': 'up' if price_variation > 0 else 'down',
                'volatility': random.uniform(0.5, 2.0)
            }
            
        return simulated_data
        
    def simulate_trade_execution(self, symbol, signal, position_size):
        """Simulate trade execution with random outcome"""
        self.trade_count += 1
        
        # Simulate trade outcome (60% win rate for demo)
        is_win = random.random() < 0.6
        profit = random.uniform(50, 200) if is_win else random.uniform(-100, -30)
        
        trade_data = {
            'symbol': symbol,
            'action': signal['action'],
            'profit': profit,
            'size': position_size,
            'timestamp': datetime.now(),
            'outcome': 'win' if is_win else 'loss'
        }
        
        # Log to FTMO system
        self.phase2_system.phase1_system.ftmo_logger.log_trade({
            'profit': profit,
            'symbol': symbol,
            'type': signal['action']
        })
        
        logging.info(f"Demo Trade {self.trade_count}: {signal['action']} {symbol} - ${profit:.2f} ({trade_data['outcome']})")
        return trade_data
        
    def run_demo(self, duration_minutes=10):
        """Run demo trading session"""
        print("🎯 FTMO PHASE 2 DEMO TRADING")
        print("=============================")
        print("Simulating market conditions and enhanced trading...")
        print(f"Demo duration: {duration_minutes} minutes")
        print("Press Ctrl+C to stop early")
        print("=============================")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                # Simulate market data
                market_data = self.simulate_market_data()
                
                # Run enhanced trading cycle
                self.phase2_system.enhanced_trading_cycle()
                
                # Simulate trades for each symbol
                for symbol in self.phase2_system.phase1_system.symbols:
                    # Simulate trading signal
                    signal_confidence = random.uniform(0.5, 0.9)
                    action = random.choice(['BUY', 'SELL', 'HOLD'])
                    
                    if action != 'HOLD':
                        # Simulate trade execution
                        position_size = random.uniform(0.1, 1.0)
                        self.simulate_trade_execution(symbol, 
                                                    {'action': action, 'confidence': signal_confidence}, 
                                                    position_size)
                
                # Display status every 5 cycles
                if self.trade_count % 5 == 0:
                    self.display_demo_status()
                
                time.sleep(10)  # 10-second cycles for demo
                
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
            
        self.display_final_results()
        
    def display_demo_status(self):
        """Display current demo status"""
        status = self.phase2_system.get_phase2_status()
        metrics = status['phase1_status']['ftmo_metrics']
        
        print(f"\n--- Demo Status (Trade #{self.trade_count}) ---")
        print(f"FTMO Progress: ${metrics['total_profit']:.2f} / ${metrics['profit_target']}")
        print(f"Daily P/L: ${metrics['daily_profit']:.2f}")
        print(f"Risk Level: {status['risk_manager']['risk_level']}")
        print(f"Drawdown: {status['drawdown_protection']['current_drawdown']:.2%}")
        print(f"Trades: {metrics['total_trades']} (Win: {metrics['winning_trades']})")
        
    def display_final_results(self):
        """Display final demo results"""
        status = self.phase2_system.get_phase2_status()
        metrics = status['phase1_status']['ftmo_metrics']
        
        print("\n🎯 DEMO TRADING RESULTS")
        print("========================")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Win Rate: {(metrics['winning_trades']/metrics['total_trades']*100 if metrics['total_trades'] > 0 else 0):.1f}%")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"FTMO Target Progress: {(metrics['total_profit']/metrics['profit_target']*100):.1f}%")
        print(f"Final Risk Level: {status['risk_manager']['risk_level']}")
        print(f"Rule Violations: {len(status['rule_enforcer']['violations_count'])}")
        print("========================")
        
        if metrics['total_profit'] >= metrics['profit_target']:
            print("🎉 FTMO TARGET ACHIEVED IN DEMO!")
        else:
            print("⚠️  FTMO target not yet reached")

def main():
    """Main demo function"""
    demo_trader = DemoTrader()
    
    # Show initial status
    initial_status = demo_trader.phase2_system.get_phase2_status()
    print("Initial System Status:")
    print(f"Account: 1600038177")
    print(f"Challenge: 200k")
    print(f"Starting Balance: ${initial_status['phase1_status']['ftmo_metrics']['current_balance']}")
    print(f"Profit Target: ${initial_status['phase1_status']['ftmo_metrics']['profit_target']}")
    
    # Run 10-minute demo
    demo_trader.run_demo(duration_minutes=10)

if __name__ == "__main__":
    main()
