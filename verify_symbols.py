print("=== VERIFYING SYMBOLS IN FILE ===")

# Read the file
try:
    with open("enhanced_trading_dashboard.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the symbols combobox line
    import re
    match = re.search(r'values=\[.*?\]', content)
    
    if match:
        print("FOUND SYMBOLS COMBOBOX:")
        print(match.group())
        print("")
        
        # Check for specific symbols
        symbols_to_check = [
            "US500Z25.sim only",
            "USOILZ25.sim only", 
            "XAUZ25.sim only",
            "US100Z25.sim only"
        ]
        
        print("MISSING SYMBOLS:")
        for symbol in symbols_to_check:
            if symbol not in match.group():
                print(f"❌ {symbol}")
            else:
                print(f"✅ {symbol}")
                
    else:
        print("❌ Could not find symbols combobox in file")
        
except Exception as e:
    print(f"Error reading file: {e}")
