import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from real_time_trading_system import RealTimeTradingSystem
from market_regime_detector import MarketRegimeDetector
import threading
import time

class UltimateTradingDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("FTMO AI Trading System - Ultimate Dashboard")
        self.master.geometry("1400x900")
        self.master.configure(bg='#2c3e50')
        
        self.trading_system = RealTimeTradingSystem()
        self.is_running = False
        self.update_thread = None
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2c3e50')
        self.style.configure('TLabel', background='#2c3e50', foreground='white', font=('Arial', 10))
        self.style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        self.style.configure('Metric.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('TButton', font=('Arial', 10))
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the ultimate trading dashboard"""
        # Main container
        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🎯 FTMO AI TRADING SYSTEM - ULTIMATE DASHBOARD", 
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        status_label = ttk.Label(header_frame, text="Status: OFFLINE", style='Metric.TLabel')
        status_label.pack(side=tk.RIGHT)
        self.status_label = status_label
        
        # Control panel
        control_frame = ttk.LabelFrame(main_container, text="System Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Initialize System", 
                  command=self.initialize_system).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Start Auto Trading", 
                  command=self.start_auto_trading).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stop Auto Trading", 
                  command=self.stop_auto_trading).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Run Single Cycle", 
                  command=self.run_single_cycle).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Show Full Report", 
                  command=self.show_full_report).pack(side=tk.LEFT, padx=2)
        
        # Main content area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Metrics and signals
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Account metrics
        metrics_frame = ttk.LabelFrame(left_panel, text="Live Account Metrics")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=50, bg='#34495e', fg='white', 
                                   font=('Consolas', 9))
        self.metrics_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Trading signals
        signals_frame = ttk.LabelFrame(left_panel, text="Current Trading Signals")
        signals_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview for signals
        columns = ('Symbol', 'Action', 'Size', 'Regime', 'RL Action', 'Confidence')
        self.signals_tree = ttk.Treeview(signals_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=100)
        
        self.signals_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Charts
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create matplotlib figures
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.patch.set_facecolor('#2c3e50')
        
        # Set dark theme for plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bottom panel - Performance history
        bottom_frame = ttk.LabelFrame(main_container, text="Performance History")
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.performance_text = tk.Text(bottom_frame, height=6, width=100, bg='#34495e', fg='white',
                                       font=('Consolas', 8))
        self.performance_text.pack(fill=tk.X, padx=5, pady=5)
    
    def initialize_system(self):
        """Initialize the trading system"""
        try:
            if self.trading_system.initialize_system():
                self.status_label.config(text="Status: INITIALIZED")
                self.update_metrics()
                messagebox.showinfo("Success", "Trading system initialized successfully!")
            else:
                messagebox.showerror("Error", "System initialization failed")
        except Exception as e:
            messagebox.showerror("Error", f"Initialization error: {str(e)}")
    
    def start_auto_trading(self):
        """Start automatic trading"""
        if not hasattr(self.trading_system, 'mt5') or not self.trading_system.mt5.connected:
            messagebox.showerror("Error", "System not initialized. Please initialize first.")
            return
        
        self.is_running = True
        self.status_label.config(text="Status: AUTO TRADING")
        
        # Start update thread
        self.update_thread = threading.Thread(target=self.auto_trading_loop, daemon=True)
        self.update_thread.start()
        
        messagebox.showinfo("Started", "Auto trading started! System will run cycles every 2 minutes.")
    
    def stop_auto_trading(self):
        """Stop automatic trading"""
        self.is_running = False
        self.status_label.config(text="Status: STOPPED")
        messagebox.showinfo("Stopped", "Auto trading stopped.")
    
    def auto_trading_loop(self):
        """Auto trading loop running in background"""
        cycle_count = 0
        while self.is_running:
            try:
                cycle_count += 1
                print(f"Auto trading cycle {cycle_count}")
                self.trading_system.run_trading_cycle()
                
                # Update GUI in main thread
                self.master.after(0, self.update_all_displays)
                
                # Wait 2 minutes between cycles
                for i in range(120):  # 120 seconds = 2 minutes
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Auto trading error: {e}")
                time.sleep(10)  # Wait before retrying
    
    def run_single_cycle(self):
        """Run a single trading cycle"""
        try:
            self.trading_system.run_trading_cycle()
            self.update_all_displays()
            messagebox.showinfo("Success", "Trading cycle completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Trading cycle failed: {str(e)}")
    
    def update_all_displays(self):
        """Update all dashboard displays"""
        self.update_metrics()
        self.update_signals()
        self.update_charts()
        self.update_performance()
    
    def update_metrics(self):
        """Update account metrics display"""
        try:
            metrics = self.trading_system.get_system_metrics()
            
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "ACCOUNT METRICS\\n")
            self.metrics_text.insert(tk.END, "=" * 40 + "\\n")
            
            account = metrics['account']
            risk = metrics['risk_management']
            rl = metrics['rl_learning']
            
            self.metrics_text.insert(tk.END, f"Balance: ${account['balance']:,.2f}\\n")
            self.metrics_text.insert(tk.END, f"Equity: ${account['equity']:,.2f}\\n")
            self.metrics_text.insert(tk.END, f"Floating P&L: ${account['floating_pnl']:,.2f}\\n")
            self.metrics_text.insert(tk.END, f"Daily P&L: ${account['daily_pnl']:,.2f}\\n")
            self.metrics_text.insert(tk.END, f"Risk Ratio: {risk['risk_ratio']:.2%}\\n")
            self.metrics_text.insert(tk.END, f"Open Positions: {risk['open_positions']}\\n")
            self.metrics_text.insert(tk.END, f"RL Success Rate: {rl['success_rate']:.2%}\\n")
            self.metrics_text.insert(tk.END, f"Total Trades: {rl['total_actions']}\\n")
            
        except Exception as e:
            self.metrics_text.insert(tk.END, f"Error updating metrics: {e}")
    
    def update_signals(self):
        """Update trading signals display"""
        try:
            # Clear existing signals
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
            # Generate new signals (without executing)
            signals = self.trading_system.generate_trading_signals()
            
            for signal in signals:
                self.signals_tree.insert('', tk.END, values=(
                    signal['symbol'],
                    signal['action'],
                    f"${signal.get('position_size', 0):.2f}",
                    signal.get('market_regime', 'Unknown'),
                    signal.get('rl_action', 'N/A'),
                    f"{signal.get('regime_confidence', 0):.2%}" if signal.get('regime_confidence') else 'N/A'
                ))
                
        except Exception as e:
            print(f"Error updating signals: {e}")
    
    def update_charts(self):
        """Update all charts"""
        try:
            # Clear existing plots
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
                ax.set_facecolor('#34495e')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
            
            metrics = self.trading_system.get_system_metrics()
            rl_metrics = metrics['rl_learning']
            
            # Chart 1: Account Balance History (simulated)
            if hasattr(self.trading_system, 'performance_log') and self.trading_system.performance_log:
                balances = [log['balance'] for log in self.trading_system.performance_log[-20:]]
                self.ax1.plot(balances, 'g-', linewidth=2)
                self.ax1.set_title('Account Balance Trend')
                self.ax1.set_ylabel('Balance ($)')
                self.ax1.grid(True, alpha=0.3)
            
            # Chart 2: RL Learning Progress
            if rl_metrics['total_actions'] > 0:
                success_rates = [0.5]  # Base rate
                if rl_metrics['success_rate'] > 0:
                    success_rates.append(rl_metrics['success_rate'])
                self.ax2.bar(['Success Rate'], success_rates, color='blue', alpha=0.7)
                self.ax2.set_title('RL Learning Performance')
                self.ax2.set_ylabel('Success Rate')
                self.ax2.set_ylim(0, 1)
            
            # Chart 3: Risk Metrics
            risk_metrics = metrics['risk_management']
            risk_data = [risk_metrics.get('risk_ratio', 0), 
                        risk_metrics.get('daily_limit_remaining', 0.05)]
            risk_labels = ['Current Risk', 'Limit Remaining']
            colors = ['red' if risk_metrics.get('risk_ratio', 0) > 0.05 else 'orange', 'green']
            
            self.ax3.bar(risk_labels, risk_data, color=colors, alpha=0.7)
            self.ax3.set_title('Risk Management')
            self.ax3.set_ylabel('Ratio')
            
            # Chart 4: Market Regime Probabilities (simulated)
            regimes = ['High Vol', 'Medium Vol', 'Low Vol']
            # Simulate regime probabilities for demo
            probs = [0.2, 0.5, 0.3]  # Example probabilities
            colors = ['red', 'yellow', 'green']
            
            self.ax4.bar(regimes, probs, color=colors, alpha=0.7)
            self.ax4.set_title('Market Regime Probabilities')
            self.ax4.set_ylabel('Probability')
            self.ax4.set_ylim(0, 1)
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Chart update error: {e}")
    
    def update_performance(self):
        """Update performance history"""
        try:
            self.performance_text.delete(1.0, tk.END)
            self.performance_text.insert(tk.END, "PERFORMANCE HISTORY\\n")
            self.performance_text.insert(tk.END, "=" * 50 + "\\n")
            
            if hasattr(self.trading_system, 'performance_log') and self.trading_system.performance_log:
                # Show last 10 performance entries
                for log in self.trading_system.performance_log[-10:]:
                    timestamp = log['timestamp'].strftime('%H:%M:%S')
                    profit = log['cycle_profit']
                    balance = log['balance']
                    self.performance_text.insert(tk.END, 
                        f"{timestamp} | Profit: ${profit:8.2f} | Balance: ${balance:,.2f} | Signals: {log['signals_generated']}\\n")
            else:
                self.performance_text.insert(tk.END, "No performance data yet. Run trading cycles to see results.\\n")
                
        except Exception as e:
            self.performance_text.insert(tk.END, f"Error: {e}")
    
    def show_full_report(self):
        """Show comprehensive system report"""
        try:
            metrics = self.trading_system.get_system_metrics()
            
            report_text = "COMPREHENSIVE SYSTEM REPORT\\n"
            report_text += "=" * 50 + "\\n\\n"
            
            # Account section
            report_text += "ACCOUNT STATUS:\\n"
            account = metrics['account']
            report_text += f"  Balance: ${account['balance']:,.2f}\\n"
            report_text += f"  Equity: ${account['equity']:,.2f}\\n"
            report_text += f"  Floating P&L: ${account['floating_pnl']:,.2f}\\n"
            report_text += f"  Daily P&L: ${account['daily_pnl']:,.2f}\\n\\n"
            
            # Risk section
            report_text += "RISK MANAGEMENT:\\n"
            risk = metrics['risk_management']
            report_text += f"  Risk Ratio: {risk['risk_ratio']:.2%}\\n"
            report_text += f"  Daily Limit Remaining: {risk['daily_limit_remaining']:.2%}\\n"
            report_text += f"  Open Positions: {risk['open_positions']}\\n\\n"
            
            # RL section
            report_text += "REINFORCEMENT LEARNING:\\n"
            rl = metrics['rl_learning']
            report_text += f"  Success Rate: {rl['success_rate']:.2%}\\n"
            report_text += f"  Total Actions: {rl['total_actions']}\\n"
            report_text += f"  Exploration Rate: {rl['exploration_rate']:.3f}\\n"
            report_text += f"  Unique States: {rl.get('unique_states', 0)}\\n\\n"
            
            # Portfolio section
            report_text += "PORTFOLIO:\\n"
            portfolio = metrics['portfolio']
            report_text += f"  Total Exposure: {portfolio['total_exposure']:.2f}\\n"
            report_text += f"  Performance History: {metrics['performance_history']} cycles\\n"
            
            messagebox.showinfo("System Report", report_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate report: {str(e)}")

def start_ultimate_dashboard():
    """Start the ultimate trading dashboard"""
    root = tk.Tk()
    app = UltimateTradingDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    start_ultimate_dashboard()
