# Test Symbol Display in Dashboard
import tkinter as tk
from tkinter import ttk

def test_symbol_display():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Test the symbols combobox
    symbols_var = tk.StringVar(value="All Optimized")
    symbols_combo = ttk.Combobox(root, textvariable=symbols_var,
                                values=["All Optimized", "BTCX25.sim only", "US30Z25.sim only", 
                                       "XAUZ25.sim only", "US100Z25.sim only", "US500Z25.sim only", 
                                       "USOILZ25.sim only"], width=15)
    
    print("Symbols combobox values:")
    for i, value in enumerate(symbols_combo['values']):
        print(f"  {i+1}. {value}")
    
    # Test the symbols mapping
    symbols_map = {
        "All Optimized": ["BTCX25.sim", "US30Z25.sim", "XAUZ25.sim", "US100Z25.sim", "US500Z25.sim", "USOILZ25.sim"],
        "BTCX25.sim only": ["BTCX25.sim"],
        "US30Z25.sim only": ["US30Z25.sim"],
        "XAUZ25.sim only": ["XAUZ25.sim"],
        "US100Z25.sim only": ["US100Z25.sim"],
        "US500Z25.sim only": ["US500Z25.sim"],
        "USOILZ25.sim only": ["USOILZ25.sim"]
    }
    
    print("\nSymbols mapping:")
    for option, symbols in symbols_map.items():
        print(f"  {option} -> {symbols}")
    
    root.destroy()
    print("\n✅ Symbol display test completed!")

if __name__ == "__main__":
    test_symbol_display()
