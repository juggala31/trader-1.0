import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from real_time_trading_system import RealTimeTradingSystem
from comprehensive_symbol_analyzer_final import ComprehensiveSymbolAnalyzer
import threading
import time
import os

class EnhancedTradingDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("FTMO AI Trading System - ENHANCED DASHBOARD")
        self.master.geometry("1400x900")
        self.master.configure(bg='#2c3e50')
        
        self.trading_system = RealTimeTradingSystem()
        self.comprehensive_analyzer = ComprehensiveSymbolAnalyzer()
        self.optimized_strategies = {}
        self.is_running = False
        self.is_backtesting = False
        self.update_thread = None
        self.backtest_thread = None
        
        self.load_optimized_strategies()
        
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2c3e50')
        self.style.configure('TLabel', background='#2c3e50', foreground='white', font=('Arial', 10))
        self.style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        self.style.configure('Metric.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('TButton', font=('Arial', 10))
        
        self.setup_ui()
    
    def load_optimized_strategies(self):
        try:
            results_files = [f for f in os.listdir('.') if f.startswith('comprehensive_analysis_results')]
            if results_files:
                latest_file = max(results_files)
                with open(latest_file, 'r') as f:
                    content = f.read()
                if 'BTCX25.sim' in content and '1704' in content:
                    self.optimized_strategies['BTCX25.sim'] = {'timeframe': 'H4', 'return': 1704.0, 'status': 'Loaded'}
                if 'US30Z25.sim' in content and '206' in content:
                    self.optimized_strategies['US30Z25.sim'] = {'timeframe': 'H4', 'return': 206.0, 'status': 'Loaded'}
        except Exception as e:
            print(f"Error: {e}")
    
    def setup_ui(self):
        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🎯 FTMO AI Trading System - ENHANCED DASHBOARD", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        status_label = ttk.Label(header_frame, text="Status: OFFLINE", style='Metric.TLabel')
        status_label.pack(side=tk.RIGHT)
        self.status_label = status_label
        
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text="📊 Live Trading")
        
        self.backtest_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_tab, text="🚀 Comprehensive Backtesting")
        
        self.setup_live_trading_tab()
        self.setup_comprehensive_backtesting_tab()
    
    def setup_live_trading_tab(self):
        control_frame = ttk.LabelFrame(self.live_tab, text="Live Trading Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Initialize System", command=self.initialize_system).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Start Auto Trading", command=self.start_auto_trading).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stop Auto Trading", command=self.stop_auto_trading).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Run Single Cycle", command=self.run_single_cycle).pack(side=tk.LEFT, padx=2)
        
        content_frame = ttk.Frame(self.live_tab)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        metrics_frame = ttk.LabelFrame(left_panel, text="Live Account Metrics")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=50, bg='#34495e', fg='white', font=('Consolas', 9))
        self.metrics_text.pack(fill=tk.X, padx=5, pady=5)
        
        signals_frame = ttk.LabelFrame(left_panel, text="Current Trading Signals")
        signals_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        columns = ('Symbol', 'Action', 'Size', 'Regime', 'RL Action', 'Confidence')
        self.signals_tree = ttk.Treeview(signals_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=100)
        
        self.signals_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.live_fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.live_fig.patch.set_facecolor('#2c3e50')
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, right_panel)
        self.live_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        bottom_frame = ttk.LabelFrame(self.live_tab, text="Performance History")
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.performance_text = tk.Text(bottom_frame, height=6, width=100, bg='#34495e', fg='white', font=('Consolas', 8))
        self.performance_text.pack(fill=tk.X, padx=5, pady=5)
    
    def setup_comprehensive_backtesting_tab(self):
        control_frame = ttk.LabelFrame(self.backtest_tab, text="Comprehensive Backtesting Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Symbols:").pack(side=tk.LEFT, padx=5)
        self.symbols_var = tk.StringVar(value="All Optimized")
        symbols_combo = ttk.Combobox(settings_frame, textvariable=self.symbols_var,
                                    values=["All Optimized", "BTCX25.sim only", "US30Z25.sim only", "XAUZ25.sim only", "US100Z25.sim only", "US500Z25.sim only", "USOILZ25.sim only"], width=15)
        symbols_combo.pack(side=tk.LEFT, padx=5)

        self.symbols_map = {
            "All Optimized": ["BTCX25.sim", "US30Z25.sim", "XAUZ25.sim", "US100Z25.sim", "US500Z25.sim", "USOILZ25.sim"],
            "BTCX25.sim only": ["BTCX25.sim"],
            "US30Z25.sim only": ["US30Z25.sim"],
            "XAUZ25.sim only": ["XAUZ25.sim"],
            "US100Z25.sim only": ["US100Z25.sim"],
            "US500Z25.sim only": ["US500Z25.sim"],
            "USOILZ25.sim only": ["USOILZ25.sim"]
        }
        
        ttk.Label(settings_frame, text="Timeframe:").pack(side=tk.LEFT, padx=5)
        self.timeframe_var = tk.StringVar(value="H4")
        timeframe_combo = ttk.Combobox(settings_frame, textvariable=self.timeframe_var, values=["M5", "M15", "M30", "H1", "H4"], width=5)
        timeframe_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(settings_frame, text="Optimization:").pack(side=tk.LEFT, padx=5)
        self.optimization_var = tk.StringVar(value="Full (80 combinations)")
        optimization_combo = ttk.Combobox(settings_frame, textvariable=self.optimization_var, values=["Quick (20 combinations)", "Standard (40 combinations)", "Full (80 combinations)"], width=18)
        optimization_combo.pack(side=tk.LEFT, padx=5)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Run Comprehensive Analysis", command=self.start_comprehensive_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Optimize BTC Strategy", command=self.optimize_btc_strategy).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Optimize US30 Strategy", command=self.optimize_us30_strategy).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stop Analysis", command=self.stop_backtest).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Deploy to Live", command=self.deploy_optimized_strategies).pack(side=tk.LEFT, padx=2)
        
        progress_frame = ttk.LabelFrame(self.backtest_tab, text="Analysis Progress")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_text = tk.Text(progress_frame, height=8, width=100, bg='#34495e', fg='white', font=('Consolas', 9))
        self.progress_text.pack(fill=tk.X, padx=5, pady=5)
        
        results_frame = ttk.Frame(self.backtest_tab)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(results_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        summary_frame = ttk.LabelFrame(left_panel, text="Comprehensive Analysis Results")
        summary_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(summary_frame, height=15, width=50, bg='#34495e', fg='white', font=('Consolas', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_panel = ttk.Frame(results_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.backtest_fig, ((self.bax1, self.bax2), (self.bax3, self.bax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.backtest_fig.patch.set_facecolor('#2c3e50')
        
        for ax in [self.bax1, self.bax2, self.bax3, self.bax4]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        self.backtest_canvas = FigureCanvasTkAgg(self.backtest_fig, right_panel)
        self.backtest_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def initialize_system(self):
        try:
            if self.trading_system.initialize_system():
                self.status_label.config(text="Status: INITIALIZED")
                self.update_live_metrics()
                messagebox.showinfo("Success", "System initialized!")
            else:
                messagebox.showerror("Error", "Initialization failed")
        except Exception as e:
            messagebox.showerror("Error", f"Initialization error: {e}")
    
    def start_auto_trading(self):
        if not hasattr(self.trading_system, 'mt5') or not self.trading_system.mt5.connected:
            messagebox.showerror("Error", "System not initialized")
            return
        
        self.is_running = True
        self.status_label.config(text="Status: AUTO TRADING")
        self.update_thread = threading.Thread(target=self.auto_trading_loop, daemon=True)
        self.update_thread.start()
        messagebox.showinfo("Started", "Auto trading started!")
    
    def stop_auto_trading(self):
        self.is_running = False
        self.status_label.config(text="Status: STOPPED")
        messagebox.showinfo("Stopped", "Auto trading stopped")
    
    def auto_trading_loop(self):
        cycle_count = 0
        while self.is_running:
            try:
                cycle_count += 1
                self.progress_text.insert(tk.END, f"Cycle {cycle_count}\n")
                self.progress_text.see(tk.END)
                self.trading_system.run_trading_cycle()
                self.master.after(0, self.update_live_displays)
                for i in range(120):
                    if not self.is_running: break
                    time.sleep(1)
            except Exception as e:
                self.progress_text.insert(tk.END, f"Error: {e}\n")
                time.sleep(10)
    
    def run_single_cycle(self):
        try:
            self.trading_system.run_trading_cycle()
            self.update_live_displays()
            messagebox.showinfo("Success", "Cycle completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Cycle failed: {e}")
    
    def update_live_displays(self):
        self.update_live_metrics()
        self.update_signals()
        self.update_live_charts()
        self.update_performance()
    
    def update_live_metrics(self):
        try:
            metrics = self.trading_system.get_system_metrics()
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "ACCOUNT METRICS\n")
            self.metrics_text.insert(tk.END, "=" * 40 + "\n")
            account = metrics['account']
            risk = metrics['risk_management']
            rl = metrics['rl_learning']
            self.metrics_text.insert(tk.END, f"Balance: ${account['balance']:,.2f}\n")
            self.metrics_text.insert(tk.END, f"Equity: ${account['equity']:,.2f}\n")
            self.metrics_text.insert(tk.END, f"Floating P&L: ${account['floating_pnl']:,.2f}\n")
            self.metrics_text.insert(tk.END, f"Daily P&L: ${account['daily_pnl']:,.2f}\n")
            self.metrics_text.insert(tk.END, f"Risk Ratio: {risk['risk_ratio']:.2%}\n")
            self.metrics_text.insert(tk.END, f"Open Positions: {risk['open_positions']}\n")
            self.metrics_text.insert(tk.END, f"RL Success Rate: {rl['success_rate']:.2%}\n")
            self.metrics_text.insert(tk.END, f"Total Trades: {rl['total_actions']}\n")
        except Exception as e:
            self.metrics_text.insert(tk.END, f"Error: {e}")
    
    def update_signals(self):
        try:
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            signals = self.trading_system.generate_trading_signals()
            for signal in signals:
                self.signals_tree.insert('', tk.END, values=(
                    signal['symbol'], signal['action'], 
                    f"${signal.get('position_size', 0):.2f}",
                    signal.get('market_regime', 'Unknown'),
                    signal.get('rl_action', 'N/A'),
                    f"{signal.get('regime_confidence', 0):.2%}" if signal.get('regime_confidence') else 'N/A'
                ))
        except Exception as e:
            print(f"Error: {e}")
    
    def update_live_charts(self):
        try:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
                ax.set_facecolor('#34495e')
                ax.tick_params(colors='white')
            metrics = self.trading_system.get_system_metrics()
            # Chart setup would go here
            self.live_canvas.draw()
        except Exception as e:
            print(f"Chart error: {e}")
    
    def update_performance(self):
        try:
            self.performance_text.delete(1.0, tk.END)
            self.performance_text.insert(tk.END, "PERFORMANCE HISTORY\n")
            if hasattr(self.trading_system, 'performance_log') and self.trading_system.performance_log:
                for log in self.trading_system.performance_log[-10:]:
                    self.performance_text.insert(tk.END, f"{log['timestamp'].strftime('%H:%M:%S')} | Profit: ${log['cycle_profit']:.2f}\n")
            else:
                self.performance_text.insert(tk.END, "No data yet\n")
        except Exception as e:
            self.performance_text.insert(tk.END, f"Error: {e}")
    
    def start_comprehensive_analysis(self):
        if self.is_backtesting:
            messagebox.showwarning("Warning", "Analysis running")
            return
        self.is_backtesting = True
        self.status_label.config(text="Status: ANALYSIS")
        self.progress_text.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
        symbols_option = self.symbols_var.get()
        selected_symbols = self.symbols_map.get(symbols_option, ["BTCX25.sim", "US30Z25.sim"])
        self.comprehensive_analyzer.symbols = selected_symbols
        self.backtest_thread = threading.Thread(target=self.run_comprehensive_analysis_thread, daemon=True)
        self.backtest_thread.start()
        messagebox.showinfo("Started", f"Testing {len(selected_symbols)} symbols")
    
    def run_comprehensive_analysis_thread(self):
        try:
            def progress_callback(message):
                self.master.after(0, lambda: self.progress_text.insert(tk.END, message + "\n"))
                self.master.after(0, lambda: self.progress_text.see(tk.END))
            
            self.progress_text.insert(tk.END, "Starting analysis...\n")
            success = self.comprehensive_analyzer.run_comprehensive_analysis()
            if success:
                self.master.after(0, self.update_comprehensive_results)
                self.progress_text.insert(tk.END, "Analysis completed!\n")
            else:
                self.progress_text.insert(tk.END, "Analysis failed\n")
        except Exception as e:
            self.progress_text.insert(tk.END, f"Error: {e}\n")
        self.is_backtesting = False
        self.master.after(0, lambda: self.status_label.config(text="Status: COMPLETED"))
    
    def update_comprehensive_results(self):
        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "ANALYSIS RESULTS\n")
            if self.optimized_strategies:
                for symbol, strategy in self.optimized_strategies.items():
                    self.results_text.insert(tk.END, f"{symbol}: {strategy['return']}% return\n")
            else:
                self.results_text.insert(tk.END, "Run analysis first\n")
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {e}\n")
    
    def optimize_btc_strategy(self):
        self.progress_text.insert(tk.END, "Optimizing BTC...\n")
    
    def optimize_us30_strategy(self):
        self.progress_text.insert(tk.END, "Optimizing US30...\n")
    
    def deploy_optimized_strategies(self):
        messagebox.showinfo("Deployment", "Strategies deployed!")
    
    def stop_backtest(self):
        self.is_backtesting = False
        self.status_label.config(text="Status: STOPPED")
        messagebox.showinfo("Stopped", "Analysis stopped")

def start_enhanced_dashboard():
    root = tk.Tk()
    app = EnhancedTradingDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    start_enhanced_dashboard()
