#!/usr/bin/env python3
"""
START.py — One-click launcher for FraudX
Run: python START.py
Then open: http://localhost:5050
"""
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DATA_DIR  = os.path.join(BASE_DIR, 'data')
ATM_CSV   = os.path.join(DATA_DIR, 'ATM_transaction_dataset.csv')
BANK_CSV  = os.path.join(DATA_DIR, 'bank_transactions_data_2.csv')

print("""
╔═══════════════════════════════════════════════════════════╗
║   🛡  FRAUDX — Banking Fraud Detection System        ║
║       Agentic AI: ATM | UPI | Credit Card | Cheque       ║
╠═══════════════════════════════════════════════════════════╣
║  URL:  http://localhost:5050                              ║
║  Stop: Ctrl+C                                            ║
╚═══════════════════════════════════════════════════════════╝
""")

# Check datasets
missing = []
for p in [ATM_CSV, BANK_CSV]:
    if not os.path.exists(p):
        missing.append(p)

if missing:
    print("❌  Missing dataset files:")
    for m in missing:
        print(f"    {m}")
    print(f"\n   → Create a 'data' folder inside {BASE_DIR}")
    print("   → Place both CSV files there:")
    print("       ATM_transaction_dataset.csv")
    print("       bank_transactions_data_2.csv")
    sys.exit(1)

print("✅  Datasets found")
print("⏳  Starting server (models load in background ~15s)...\n")

os.chdir(BASE_DIR)
from app import app
app.run(host='0.0.0.0', port=5050, debug=False, threaded=True, use_reloader=False)
