# FTMO Tkinter GUI Dashboard - No External Dependencies
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from datetime import datetime
from ftmo_minimal import FTMOMinimalSystem

class FTMO_Tkinter_Dashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FTMO Trading Dashboard - 200k Challenge")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2C3E50')
        
        self.trading_system = FTMOMinimalSystem("1600038177", "200k", live_mode=False)
        self.trading_active = False
        self.trading_thread = None
        
        self.setup_gui()
        self.start_updates()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Header
        header_frame = tk.Frame(self.root, bg='#34495E', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="FTMO 200K CHALLENGE DASHBOARD", 
                              font=('Arial', 20, 'bold'), fg='#ECF0F1', bg='#34495E')
        title_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left column - Challenge Progress
        left_frame = tk.Frame(main_frame, bg='#2C3E50')
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.create_challenge_section(left_frame)
        self.create_metrics_section(left_frame)
        
        # Right column - Trades and Log
        right_frame = tk.Frame(main_frame, bg='#2C3E50')
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        self.create_trades_section(right_frame)
        self.create_log_section(right_frame)
        
        # Control buttons
        control_frame = tk.Frame(self.root, bg='#2C3E50')
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_btn = tk.Button(control_frame, text="START TRADING", 
                                  command=self.start_trading, bg='#27AE60', fg='white',
                                  font=('Arial', 12, 'bold'), width=15)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="STOP TRADING", 
                                 command=self.stop_trading, bg='#E74C3C', fg='white',
                                 font=('Arial', 12, 'bold'), width=15, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
    def create_challenge_section(self, parent):
        """Create challenge progress section"""
        challenge_frame = ttk.LabelFrame(parent, text="Challenge Progress", padding=10)
        challenge_frame.pack(fill='x', pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(challenge_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        
        # Progress labels
        progress_labels = tk.Frame(challenge_frame)
        progress_labels.pack(fill='x')
        
        self.current_profit_label = tk.Label(progress_labels, text="Current Profit: $0.00", 
                                            font=('Arial', 10), bg='white')
        self.current_profit_label.pack(side='left', expand=True)
        
        self.target_label = tk.Label(progress_labels, text="Target: $10,000.00", 
                                    font=('Arial', 10), bg='white')
        self.target_label.pack(side='left', expand=True)
        
        self.progress_percent_label = tk.Label(progress_labels, text="0.0% Complete", 
                                             font=('Arial', 10, 'bold'), bg='white')
        self.progress_percent_label.pack(side='right')
        
    def create_metrics_section(self, parent):
        """Create trading metrics section"""
        metrics_frame = ttk.LabelFrame(parent, text="Trading Metrics", padding=10)
        metrics_frame.pack(fill='x', pady=5)
        
        # Strategy info
        strategy_frame = tk.Frame(metrics_frame)
        strategy_frame.pack(fill='x', pady=2)
        
        tk.Label(strategy_frame, text="Current Strategy:", font=('Arial', 9), bg='white').pack(side='left')
        self.strategy_label = tk.Label(strategy_frame, text="XGBoost", font=('Arial', 9, 'bold'), bg='white')
        self.strategy_label.pack(side='left', padx=5)
        
        # Metrics grid
        metrics_grid = tk.Frame(metrics_frame)
        metrics_grid.pack(fill='x', pady=5)
        
        # Row 1
        tk.Label(metrics_grid, text="Win Rate:", bg='white').grid(row=0, column=0, sticky='w')
        self.win_rate_label = tk.Label(metrics_grid, text="0.0%", bg='white')
        self.win_rate_label.grid(row=0, column=1, sticky='w', padx=10)
        
        tk.Label(metrics_grid, text="Total Trades:", bg='white').grid(row=0, column=2, sticky='w')
        self.total_trades_label = tk.Label(metrics_grid, text="0", bg='white')
        self.total_trades_label.grid(row=0, column=3, sticky='w', padx=10)
        
        # Row 2
        tk.Label(metrics_grid, text="Daily P/L:", bg='white').grid(row=1, column=0, sticky='w')
        self.daily_profit_label = tk.Label(metrics_grid, text="$0.00", bg='white')
        self.daily_profit_label.grid(row=1, column=1, sticky='w', padx=10)
        
        tk.Label(metrics_grid, text="Active Trades:", bg='white').grid(row=1, column=2, sticky='w')
        self.active_trades_label = tk.Label(metrics_grid, text="0", bg='white')
        self.active_trades_label.grid(row=1, column=3, sticky='w', padx=10)
        
    def create_trades_section(self, parent):
        """Create recent trades section"""
        trades_frame = ttk.LabelFrame(parent, text="Recent Trades", padding=10)
        trades_frame.pack(fill='both', expand=True, pady=5)
        
        # Create treeview for trades
        columns = ('Time', 'Symbol', 'Type', 'Size', 'P/L', 'Status')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='tree headings', height=8)
        
        # Configure columns
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=80)
        
        self.trades_tree.pack(fill='both', expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(trades_frame, orient='vertical', command=self.trades_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.trades_tree.configure(yscrollcommand=scrollbar.set)
        
    def create_log_section(self, parent):
        """Create system log section"""
        log_frame = ttk.LabelFrame(parent, text="System Log", padding=10)
        log_frame.pack(fill='both', expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True)
        self.log_text.configure(state='disabled')
        
    def start_updates(self):
        """Start periodic GUI updates"""
        def update_loop():
            while True:
                self.update_dashboard()
                time.sleep(2)  # Update every 2 seconds
                
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
    def update_dashboard(self):
        """Update all dashboard elements"""
        try:
            status = self.trading_system.get_system_status()
            self.update_challenge_progress(status)
            self.update_trading_metrics(status)
            self.update_recent_trades()
            self.update_system_log()
            
        except Exception as e:
            self.log_message(f"Error updating dashboard: {e}")
            
    def update_challenge_progress(self, status):
        """Update challenge progress"""
        progress = status.get('progress', 0)
        current_profit = progress * 10000 / 100
        
        self.progress_var.set(progress)
        self.current_profit_label.config(text=f"Current Profit: ${current_profit:,.2f}")
        self.progress_percent_label.config(text=f"{progress:.1f}% Complete")
        
        # Update progress bar color based on progress
        if progress >= 80:
            self.progress_bar.configure(style='Green.Horizontal.TProgressbar')
        elif progress >= 50:
            self.progress_bar.configure(style='Yellow.Horizontal.TProgressbar')
        else:
            self.progress_bar.configure(style='Red.Horizontal.TProgressbar')
            
    def update_trading_metrics(self, status):
        """Update trading metrics"""
        self.strategy_label.config(text=status.get('strategy', 'XGBoost'))
        
        # Sample metrics (would come from real data)
        self.win_rate_label.config(text="65.5%")
        self.total_trades_label.config(text="12")
        self.daily_profit_label.config(text="$450.50")
        self.active_trades_label.config(text="2")
        
    def update_recent_trades(self):
        """Update recent trades table"""
        # Clear existing items
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
            
        # Sample trade data
        sample_trades = [
            ("14:30:05", "US30", "BUY", "0.5", "+$125.50", "WIN"),
            ("14:45:22", "US100", "SELL", "0.3", "-$45.25", "LOSS"),
            ("15:10:18", "UAX", "BUY", "0.4", "+$87.30", "WIN")
        ]
        
        for trade in sample_trades:
            self.trades_tree.insert('', 'end', values=trade)
            
    def update_system_log(self):
        """Update system log"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Sample log entries
        log_entries = [
            f"{current_time} - Market analysis complete",
            f"{current_time} - XGBoost signal: BUY US30 (72% confidence)",
            f"{current_time} - Risk validation passed",
            f"{current_time} - Trade executed: BUY US30 0.5 lots"
        ]
        
        # Add new entry occasionally
        import random
        if random.random() < 0.1:  # 10% chance to add entry
            self.log_message(random.choice(log_entries))
            
    def log_message(self, message):
        """Add message to log"""
        self.log_text.configure(state='normal')
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')
        
    def start_trading(self):
        """Start trading"""
        self.trading_active = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.log_message("Trading started...")
        
        # Start trading in background thread
        self.trading_thread = threading.Thread(target=self.run_trading, daemon=True)
        self.trading_thread.start()
        
    def run_trading(self):
        """Run trading simulation"""
        while self.trading_active:
            try:
                # Simulate trading activity
                time.sleep(5)
            except Exception as e:
                self.log_message(f"Trading error: {e}")
                time.sleep(10)
                
    def stop_trading(self):
        """Stop trading"""
        self.trading_active = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log_message("Trading stopped...")
        
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def main():
    """Main function to launch the GUI"""
    print("🚀 Starting FTMO Tkinter Dashboard...")
    dashboard = FTMO_Tkinter_Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
