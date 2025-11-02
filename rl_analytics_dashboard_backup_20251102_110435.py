import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from rl_enhanced_system import RLEnhancedTradingSystem

class RLAnalyticsDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("FTMO Trading System - RL Analytics")
        self.master.geometry("1200x800")
        
        self.system = RLEnhancedTradingSystem()
        self.cycle_count = 0
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Reinforcement Learning Trading Analytics", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        ttk.Button(control_frame, text="Initialize System", 
                  command=self.initialize_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Run Trading Cycle", 
                  command=self.run_trading_cycle).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Show Metrics", 
                  command=self.show_metrics).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset Learning", 
                  command=self.reset_learning).pack(side=tk.LEFT, padx=5)
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(main_frame, text="Live Metrics")
        metrics_frame.pack(fill=tk.X, pady=10)
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=100)
        self.metrics_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Charts frame
        charts_frame = ttk.Frame(main_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create matplotlib figures
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def initialize_system(self):
        """Initialize the RL trading system"""
        try:
            self.system.initialize_system()
            self.cycle_count = 0
            self.update_metrics()
            messagebox.showinfo("Success", "RL trading system initialized!")
        except Exception as e:
            messagebox.showerror("Error", f"Initialization failed: {str(e)}")
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        try:
            self.system.run_trading_cycle()
            self.cycle_count += 1
            self.update_metrics()
            self.update_charts()
            messagebox.showinfo("Success", f"Trading cycle {self.cycle_count} completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Trading cycle failed: {str(e)}")
    
    def update_metrics(self):
        """Update the metrics display"""
        metrics = self.system.get_system_metrics()
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "REINFORCEMENT LEARNING METRICS\\n")
        self.metrics_text.insert(tk.END, "=" * 50 + "\\n\\n")
        
        # RL Metrics
        rl_metrics = metrics["rl_learning"]
        self.metrics_text.insert(tk.END, "🤖 RL Learning Metrics:\\n")
        self.metrics_text.insert(tk.END, f"  Total Reward: {rl_metrics['total_reward']:.2f}\\n")
        self.metrics_text.insert(tk.END, f"  Average Reward: {rl_metrics['avg_reward']:.2f}\\n")
        self.metrics_text.insert(tk.END, f"  Success Rate: {rl_metrics['success_rate']:.2%}\\n")
        self.metrics_text.insert(tk.END, f"  Exploration Rate: {rl_metrics['exploration_rate']:.3f}\\n")
        self.metrics_text.insert(tk.END, f"  Total Actions: {rl_metrics['total_actions']}\\n\\n")
        
        # Market Metrics
        market_metrics = metrics["market_regimes"]
        self.metrics_text.insert(tk.END, "📊 Market Regime Metrics:\\n")
        self.metrics_text.insert(tk.END, f"  Current Regime: {market_metrics['current_regime']}\\n")
        self.metrics_text.insert(tk.END, f"  Confidence: {market_metrics['confidence']:.2%}\\n\\n")
        
        # Position Metrics
        positions = metrics["current_positions"]
        self.metrics_text.insert(tk.END, "💰 Current Positions:\\n")
        for symbol, position in positions.items():
            self.metrics_text.insert(tk.END, f"  {symbol}: {position}\\n")
        
        self.metrics_text.insert(tk.END, f"\\n🔄 Trading Cycles: {self.cycle_count}")
    
    def update_charts(self):
        """Update the matplotlib charts"""
        try:
            # Clear existing plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Get RL agent data
            rl_agent = self.system.rl_trader.rl_agent
            
            # Chart 1: Reward History
            if rl_agent.reward_history:
                self.ax1.plot(rl_agent.reward_history[-100:], 'b-', alpha=0.7)
                self.ax1.set_title('Recent Rewards')
                self.ax1.set_ylabel('Reward')
                self.ax1.grid(True, alpha=0.3)
            
            # Chart 2: Exploration Rate
            exploration_rates = [0.3 * (0.995 ** i) for i in range(self.cycle_count + 1)]
            self.ax2.plot(exploration_rates, 'r-')
            self.ax2.set_title('Exploration Rate Decay')
            self.ax2.set_ylabel('Exploration Rate')
            self.ax2.grid(True, alpha=0.3)
            
            # Chart 3: Action Distribution
            if rl_agent.action_history:
                action_counts = {action: rl_agent.action_history.count(action) for action in rl_agent.actions}
                actions = list(action_counts.keys())
                counts = list(action_counts.values())
                
                self.ax3.bar(actions, counts, color='skyblue', alpha=0.7)
                self.ax3.set_title('Action Distribution')
                self.ax3.set_ylabel('Count')
                self.ax3.tick_params(axis='x', rotation=45)
            
            # Chart 4: Q-Table Size
            q_size = len(rl_agent.q_table) if rl_agent.q_table else 0
            self.ax4.bar(['Q-Table'], [q_size], color='green', alpha=0.7)
            self.ax4.set_title('Learning Progress (Q-Table Size)')
            self.ax4.set_ylabel('Number of States')
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Chart update error: {e}")
    
    def show_metrics(self):
        """Show detailed metrics in message box"""
        metrics = self.system.get_system_metrics()
        
        metrics_text = "DETAILED SYSTEM METRICS\\n"
        metrics_text += "=" * 30 + "\\n"
        
        for category, data in metrics.items():
            metrics_text += f"\\n{category.upper()}:\\n"
            if isinstance(data, dict):
                for key, value in data.items():
                    metrics_text += f"  {key}: {value}\\n"
            else:
                metrics_text += f"  {data}\\n"
        
        messagebox.showinfo("Detailed Metrics", metrics_text)
    
    def reset_learning(self):
        """Reset the RL learning"""
        self.system = RLEnhancedTradingSystem()
        self.cycle_count = 0
        self.update_metrics()
        messagebox.showinfo("Success", "RL learning reset!")

def start_rl_dashboard():
    """Start the RL analytics dashboard"""
    root = tk.Tk()
    app = RLAnalyticsDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    start_rl_dashboard()
