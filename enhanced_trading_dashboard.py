import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from real_time_trading_system import RealTimeTradingSystem
from working_backtester import WorkingBacktester  # CHANGED: Use working backtester
import threading
import time
import os

class EnhancedTradingDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("FTMO AI Trading System - Enhanced Dashboard")
        self.master.geometry("1400x900")
        self.master.configure(bg='#2c3e50')
        
        self.trading_system = RealTimeTradingSystem()
        self.backtester = WorkingBacktester()  # CHANGED: Use working backtester
        self.is_running = False
        self.is_backtesting = False
        self.update_thread = None
        self.backtest_thread = None
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2c3e50')
        self.style.configure('TLabel', background='#2c3e50', foreground='white', font=('Arial', 10))
        self.style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        self.style.configure('Metric.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('TButton', font=('Arial', 10))
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the enhanced trading dashboard with backtesting tab"""
        # Main container
        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🎯 FTMO AI TRADING SYSTEM - ENHANCED DASHBOARD", 
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        status_label = ttk.Label(header_frame, text="Status: OFFLINE", style='Metric.TLabel')
        status_label.pack(side=tk.RIGHT)
        self.status_label = status_label
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Live Trading Tab
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text="📊 Live Trading")
        
        # Backtesting Tab
        self.backtest_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_tab, text="📈 Backtesting")
        
        # Setup both tabs
        self.setup_live_trading_tab()
        self.setup_backtesting_tab()
    
    def setup_live_trading_tab(self):
        """Setup the live trading tab (existing functionality)"""
        # Control panel
        control_frame = ttk.LabelFrame(self.live_tab, text="Live Trading Controls")
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
        
        # Metrics and signals frame
        content_frame = ttk.Frame(self.live_tab)
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
        
        # Create matplotlib figures for live trading
        self.live_fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.live_fig.patch.set_facecolor('#2c3e50')
        
        # Set dark theme for plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, right_panel)
        self.live_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Performance history
        bottom_frame = ttk.LabelFrame(self.live_tab, text="Performance History")
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.performance_text = tk.Text(bottom_frame, height=6, width=100, bg='#34495e', fg='white',
                                       font=('Consolas', 8))
        self.performance_text.pack(fill=tk.X, padx=5, pady=5)
    
    def setup_backtesting_tab(self):
        """Setup the backtesting tab with controls and results"""
        # Backtesting controls
        control_frame = ttk.LabelFrame(self.backtest_tab, text="Backtesting Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Control buttons and settings
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Years selection
        ttk.Label(settings_frame, text="Years of Data:").pack(side=tk.LEFT, padx=5)
        self.years_var = tk.StringVar(value="1")
        years_combo = ttk.Combobox(settings_frame, textvariable=self.years_var, 
                                  values=["1", "2"], width=5)
        years_combo.pack(side=tk.LEFT, padx=5)
        
        # Symbols selection
        ttk.Label(settings_frame, text="Symbols:").pack(side=tk.LEFT, padx=5)
        self.symbols_var = tk.StringVar(value="All")
        symbols_combo = ttk.Combobox(settings_frame, textvariable=self.symbols_var,
                                    values=["All", "US30 only", "US100 only", "XAU only"], width=10)
        symbols_combo.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Run Backtest", 
                  command=self.start_backtest).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stop Backtest", 
                  command=self.stop_backtest).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Load Previous Results", 
                  command=self.load_backtest_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Generate Report", 
                  command=self.generate_backtest_report).pack(side=tk.LEFT, padx=2)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.backtest_tab, text="Backtesting Progress")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_text = tk.Text(progress_frame, height=6, width=100, bg='#34495e', fg='white',
                                    font=('Consolas', 9))
        self.progress_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Results frame
        results_frame = ttk.Frame(self.backtest_tab)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Results summary
        left_panel = ttk.Frame(results_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Results summary
        summary_frame = ttk.LabelFrame(left_panel, text="Backtest Results Summary")
        summary_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(summary_frame, height=15, width=50, bg='#34495e', fg='white',
                                   font=('Consolas', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Backtesting charts
        right_panel = ttk.Frame(results_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create matplotlib figures for backtesting
        self.backtest_fig, ((self.bax1, self.bax2), (self.bax3, self.bax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.backtest_fig.patch.set_facecolor('#2c3e50')
        
        # Set dark theme for plots
        for ax in [self.bax1, self.bax2, self.bax3, self.bax4]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        self.backtest_canvas = FigureCanvasTkAgg(self.backtest_fig, right_panel)
        self.backtest_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Live Trading Methods (existing functionality)
    def initialize_system(self):
        """Initialize the trading system"""
        try:
            if self.trading_system.initialize_system():
                self.status_label.config(text="Status: INITIALIZED")
                self.update_live_metrics()
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
                self.progress_text.insert(tk.END, f"Auto trading cycle {cycle_count}\\n")
                self.progress_text.see(tk.END)
                
                self.trading_system.run_trading_cycle()
                
                # Update GUI in main thread
                self.master.after(0, self.update_live_displays)
                
                # Wait 2 minutes between cycles
                for i in range(120):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.progress_text.insert(tk.END, f"Auto trading error: {e}\\n")
                time.sleep(10)
    
    def run_single_cycle(self):
        """Run a single trading cycle"""
        try:
            self.trading_system.run_trading_cycle()
            self.update_live_displays()
            messagebox.showinfo("Success", "Trading cycle completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Trading cycle failed: {str(e)}")
    
    def update_live_displays(self):
        """Update all live trading displays"""
        self.update_live_metrics()
        self.update_signals()
        self.update_live_charts()
        self.update_performance()
    
    def update_live_metrics(self):
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
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
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
    
    def update_live_charts(self):
        """Update live trading charts"""
        try:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
                ax.set_facecolor('#34495e')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
            
            metrics = self.trading_system.get_system_metrics()
            rl_metrics = metrics['rl_learning']
            
            # Chart 1: Account Balance History
            if hasattr(self.trading_system, 'performance_log') and self.trading_system.performance_log:
                balances = [log['balance'] for log in self.trading_system.performance_log[-20:]]
                self.ax1.plot(balances, 'g-', linewidth=2)
                self.ax1.set_title('Account Balance Trend')
                self.ax1.set_ylabel('Balance ($)')
                self.ax1.grid(True, alpha=0.3)
            
            # Chart 2: RL Learning Progress
            if rl_metrics['total_actions'] > 0:
                success_rates = [0.5, rl_metrics['success_rate']]
                self.ax2.bar(['Base', 'Current'], success_rates, color=['gray', 'blue'], alpha=0.7)
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
            
            # Chart 4: Market Regime Probabilities
            regimes = ['High Vol', 'Medium Vol', 'Low Vol']
            probs = [0.2, 0.5, 0.3]  # Example probabilities
            colors = ['red', 'yellow', 'green']
            
            self.ax4.bar(regimes, probs, color=colors, alpha=0.7)
            self.ax4.set_title('Market Regime Probabilities')
            self.ax4.set_ylabel('Probability')
            self.ax4.set_ylim(0, 1)
            
            self.live_canvas.draw()
            
        except Exception as e:
            print(f"Live chart update error: {e}")
    
    def update_performance(self):
        """Update performance history"""
        try:
            self.performance_text.delete(1.0, tk.END)
            self.performance_text.insert(tk.END, "PERFORMANCE HISTORY\\n")
            self.performance_text.insert(tk.END, "=" * 50 + "\\n")
            
            if hasattr(self.trading_system, 'performance_log') and self.trading_system.performance_log:
                for log in self.trading_system.performance_log[-10:]:
                    timestamp = log['timestamp'].strftime('%H:%M:%S')
                    profit = log['cycle_profit']
                    balance = log['balance']
                    self.performance_text.insert(tk.END, 
                        f"{timestamp} | Profit: ${profit:8.2f} | Balance: ${balance:,.2f} | Signals: {log['signals_generated']}\\n")
            else:
                self.performance_text.insert(tk.END, "No performance data yet.\\n")
                
        except Exception as e:
            self.performance_text.insert(tk.END, f"Error: {e}")
    
    # Backtesting Methods - UPDATED for working backtester
    def start_backtest(self):
        """Start backtesting in a separate thread"""
        if self.is_backtesting:
            messagebox.showwarning("Warning", "Backtest already running")
            return
        
        self.is_backtesting = True
        self.status_label.config(text="Status: BACKTESTING")
        
        # Clear previous results
        self.progress_text.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
        
        # Get settings
        years = int(self.years_var.get())
        symbols_option = self.symbols_var.get()
        
        # Map symbols option to actual symbols
        symbols_map = {
            "All": ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"],
            "US30 only": ["US30Z25.sim"],
            "US100 only": ["US100Z25.sim"],
            "XAU only": ["XAUZ25.sim"]
        }
        
        symbols = symbols_map.get(symbols_option, ["US30Z25.sim"])
        self.backtester.symbols = symbols
        
        # Start backtest thread
        self.backtest_thread = threading.Thread(
            target=self.run_backtest_thread, 
            args=(years,),
            daemon=True
        )
        self.backtest_thread.start()
        
        messagebox.showinfo("Started", f"Backtesting started with {years} year(s) of data")
    
    def stop_backtest(self):
        """Stop backtesting"""
        self.is_backtesting = False
        self.status_label.config(text="Status: BACKTEST STOPPED")
        messagebox.showinfo("Stopped", "Backtesting stopped")
    
    def run_backtest_thread(self, years):
        """Run backtesting in background thread using WORKING backtester"""
        try:
            # Set progress callback
            def progress_callback(message):
                self.master.after(0, lambda: self.progress_text.insert(tk.END, message + "\\n"))
                self.master.after(0, lambda: self.progress_text.see(tk.END))
            
            self.backtester.set_progress_callback(progress_callback)
            
            self.progress_text.insert(tk.END, f"Starting backtest with {years} year(s) of data\\n")
            self.progress_text.see(tk.END)
            
            # Use the working backtester's method
            success = self.backtester.run_comprehensive_backtest(years=years, save_models=True)
            
            if success:
                self.master.after(0, self.update_backtest_results)
                self.progress_text.insert(tk.END, "Backtesting completed successfully!\\n")
            else:
                self.progress_text.insert(tk.END, "Backtesting completed with limited results\\n")
                
        except Exception as e:
            self.progress_text.insert(tk.END, f"Backtesting error: {e}\\n")
        
        self.is_backtesting = False
        self.master.after(0, lambda: self.status_label.config(text="Status: BACKTEST COMPLETED"))
    
    def update_backtest_results(self):
        """Update backtesting results display"""
        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "BACKTEST RESULTS\\n")
            self.results_text.insert(tk.END, "=" * 50 + "\\n\\n")
            
            if hasattr(self.backtester, 'backtest_results') and self.backtester.backtest_results:
                for symbol, results in self.backtester.backtest_results.items():
                    self.results_text.insert(tk.END, f"{symbol}:\\n")
                    self.results_text.insert(tk.END, f"  Accuracy: {results['accuracy']:.3f}\\n")
                    self.results_text.insert(tk.END, f"  Return: {results['total_return']:.2%}\\n")
                    self.results_text.insert(tk.END, f"  Trades: {results['total_trades']}\\n")
                    self.results_text.insert(tk.END, f"  Samples: {results['data_points']}\\n\\n")
                
                # Update backtesting charts
                self.update_backtest_charts()
                
            else:
                self.results_text.insert(tk.END, "No backtest results available\\n")
                
        except Exception as e:
            self.results_text.insert(tk.END, f"Error updating results: {e}\\n")
    
    def update_backtest_charts(self):
        """Update backtesting charts"""
        try:
            for ax in [self.bax1, self.bax2, self.bax3, self.bax4]:
                ax.clear()
                ax.set_facecolor('#34495e')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
            
            if hasattr(self.backtester, 'backtest_results') and self.backtester.backtest_results:
                results = self.backtester.backtest_results
                symbols = list(results.keys())
                accuracies = [results[symbol]['accuracy'] for symbol in symbols]
                returns = [results[symbol]['total_return'] for symbol in symbols]
                
                # Chart 1: Accuracy by Symbol
                self.bax1.bar(symbols, accuracies, color='blue', alpha=0.7)
                self.bax1.set_title('Model Accuracy by Symbol')
                self.bax1.set_ylabel('Accuracy')
                self.bax1.set_ylim(0, 1)
                self.bax1.tick_params(axis='x', rotation=45)
                
                # Chart 2: Returns by Symbol
                self.bax2.bar(symbols, returns, color='green', alpha=0.7)
                self.bax2.set_title('Total Returns by Symbol')
                self.bax2.set_ylabel('Return (%)')
                self.bax2.tick_params(axis='x', rotation=45)
                
                # Chart 3: Trade Counts
                trade_counts = [results[symbol]['total_trades'] for symbol in symbols]
                self.bax3.bar(symbols, trade_counts, color='orange', alpha=0.7)
                self.bax3.set_title('Trade Count by Symbol')
                self.bax3.set_ylabel('Number of Trades')
                self.bax3.tick_params(axis='x', rotation=45)
                
                # Chart 4: Data Points
                data_points = [results[symbol]['data_points'] for symbol in symbols]
                self.bax4.bar(symbols, data_points, color='purple', alpha=0.7)
                self.bax4.set_title('Data Points by Symbol')
                self.bax4.set_ylabel('Number of Bars')
                self.bax4.tick_params(axis='x', rotation=45)
            
            self.backtest_canvas.draw()
            
        except Exception as e:
            print(f"Backtest chart update error: {e}")
    
    def load_backtest_results(self):
        """Load previous backtest results from file"""
        try:
            if os.path.exists("backtesting_report.txt"):
                with open("backtesting_report.txt", 'r') as f:
                    content = f.read()
                
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "PREVIOUS BACKTEST RESULTS\\n")
                self.results_text.insert(tk.END, "=" * 50 + "\\n\\n")
                self.results_text.insert(tk.END, content)
                
                messagebox.showinfo("Success", "Previous results loaded")
            else:
                messagebox.showwarning("Warning", "No previous results found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not load results: {str(e)}")
    
    def generate_backtest_report(self):
        """Generate and save backtest report"""
        try:
            if hasattr(self.backtester, 'backtest_results') and self.backtester.backtest_results:
                # The working backtester already generates reports automatically
                messagebox.showinfo("Success", "Backtest report already generated: backtesting_report.txt")
            else:
                messagebox.showwarning("Warning", "No backtest results to report")
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate report: {str(e)}")

def start_enhanced_dashboard():
    """Start the enhanced trading dashboard"""
    root = tk.Tk()
    app = EnhancedTradingDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    start_enhanced_dashboard()
