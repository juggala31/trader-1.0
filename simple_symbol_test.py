print("=== SYMBOL VERIFICATION TEST ===")

# Test the expected symbols
expected_symbols = ["BTCX25.sim only", "US30Z25.sim only", "XAUZ25.sim only", "US100Z25.sim only", "US500Z25.sim only", "USOILZ25.sim only"]

print("Expected symbols in dropdown:")
for i, symbol in enumerate(expected_symbols, 1):
    print(f"  {i}. {symbol}")

print("")
print("✅ If these appear in your dashboard, the fix worked!")
print("🚀 Launch with: start_enhanced_dashboard.bat")
