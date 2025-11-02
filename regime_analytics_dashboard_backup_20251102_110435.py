import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from market_regime_detector import MarketRegimeDetector, RegimeAwareTrading

class SimpleRegimeSystem:
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.regime_trader = RegimeAwareTrading(self.regime_detector)
        self.regime_history = []
        self.symbols = ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]
    
    def initialize(self):
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
        price_series = pd.Series(prices, index=dates)
        self.regime_detector.train_model(price_series)
        return True
    
    def get_current_regime(self):
        dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        price_series = pd.Series(prices, index=dates)
        
        regime_pred = self.regime_detector.predict_regime(price_series)
        self.regime_history.append(regime_pred)
        return regime_pred
    
    def generate_signals(self):
        regime_pred = self.get_current_regime()
        
        signals = []
        for symbol in self.symbols:
            signal = {
                "symbol": symbol,
                "direction": "BUY" if np.random.random() > 0.5 else "SELL",
                "strength": np.random.uniform(0.5, 0.9),
                "position_size": 1.0
            }
            adapted_signal = self.regime_trader.adapt_signals(signal, regime_pred)
            signals.append(adapted_signal)
        
        return signals

class RegimeAnalyticsDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("FTMO Trading System - Market Regime Analytics")
        self.master.geometry("1000x700")
        
        self.system = SimpleRegimeSystem()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Market Regime Analytics Dashboard", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        ttk.Button(control_frame, text="Initialize System", 
                  command=self.initialize_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Update Analysis", 
                  command=self.update_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate Signals", 
                  command=self.generate_signals).pack(side=tk.LEFT, padx=5)
        
        # Status frame
        self.status_frame = ttk.LabelFrame(main_frame, text="Current Market Regime")
        self.status_frame.pack(fill=tk.X, pady=10)
        
        self.regime_label = ttk.Label(self.status_frame, text="Regime: Not Analyzed", 
                                     font=('Arial', 12))
        self.regime_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(self.status_frame, text="Confidence: -")
        self.confidence_label.pack(pady=2)
        
        # Signals frame
        self.signals_frame = ttk.LabelFrame(main_frame, text="Trading Signals")
        self.signals_frame.pack(fill=tk.X, pady=10)
        
        self.signals_text = tk.Text(self.signals_frame, height=6, width=80)
        self.signals_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Chart frame
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Regime history frame
        history_frame = ttk.LabelFrame(main_frame, text="Regime History (Last 10)")
        history_frame.pack(fill=tk.X, pady=10)
        
        # Treeview for regime history
        columns = ('Timestamp', 'Regime', 'Label', 'Confidence')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120)
        
        self.history_tree.pack(fill=tk.X, padx=5, pady=5)
    
    def initialize_system(self):
        """Initialize the regime detection system"""
        try:
            self.system.initialize()
            messagebox.showinfo("Success", "Market regime system initialized successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize system: {str(e)}")
    
    def update_analysis(self):
        """Update the current regime analysis"""
        try:
            regime_analysis = self.system.get_current_regime()
            
            # Update status labels
            self.regime_label.config(text=f"Regime: {regime_analysis['regime_label']}")
            self.confidence_label.config(text=f"Confidence: {regime_analysis['confidence']:.2%}")
            
            # Update regime history
            self.update_history_tree()
            
            # Update chart
            self.update_chart()
            
            messagebox.showinfo("Success", "Regime analysis updated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update analysis: {str(e)}")
    
    def generate_signals(self):
        """Generate and display trading signals"""
        try:
            signals = self.system.generate_signals()
            
            self.signals_text.delete(1.0, tk.END)
            self.signals_text.insert(tk.END, "Regime-Enhanced Trading Signals:\n\n")
            
            for signal in signals:
                self.signals_text.insert(tk.END, f"Symbol: {signal['symbol']}\n")
                self.signals_text.insert(tk.END, f"Direction: {signal['direction']}\n")
                self.signals_text.insert(tk.END, f"Regime: {signal['market_regime']}\n")
                self.signals_text.insert(tk.END, f"Confidence: {signal['regime_confidence']:.2%}\n")
                self.signals_text.insert(tk.END, f"Aggressiveness: {signal['regime_parameters']['aggressiveness']}\n")
                self.signals_text.insert(tk.END, f"Position Size Multiplier: {signal['regime_parameters']['position_size_multiplier']}\n")
                self.signals_text.insert(tk.END, "-" * 40 + "\n")
            
            messagebox.showinfo("Success", f"Generated {len(signals)} trading signals!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate signals: {str(e)}")
    
    def update_history_tree(self):
        """Update the regime history treeview"""
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Add new items from regime history (last 10)
        for record in self.system.regime_history[-10:]:
            self.history_tree.insert('', tk.END, values=(
                pd.Timestamp.now().strftime('%H:%M:%S'),
                record['regime'],
                record['regime_label'],
                f"{record['confidence']:.2%}"
            ))
    
    def update_chart(self):
        """Update the matplotlib chart"""
        try:
            self.ax.clear()
            
            # Create sample regime data for visualization
            regimes = ['High Volatility', 'Medium Volatility', 'Low Volatility']
            
            if self.system.regime_history:
                current_regime = self.system.regime_history[-1]['regime']
                probabilities = [0.1, 0.1, 0.1]  # Base probabilities
                probabilities[current_regime] = 0.7  # Highlight current regime
            else:
                probabilities = [0.33, 0.34, 0.33]
            
            colors = ['red', 'yellow', 'green']
            bars = self.ax.bar(regimes, probabilities, color=colors, alpha=0.7)
            
            self.ax.set_title('Current Regime Probabilities')
            self.ax.set_ylabel('Probability')
            self.ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{prob:.1%}', ha='center', va='bottom')
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Chart update error: {e}")

def start_dashboard():
    """Start the regime analytics dashboard"""
    root = tk.Tk()
    app = RegimeAnalyticsDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    start_dashboard()
