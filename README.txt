╔══════════════════════════════════════════════════════════╗
║   🛡  FraudX — Banking Fraud Detection System       ║
╚══════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 1 — Install Python 3.9+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Download from: https://python.org
  ✅ Check "Add to PATH" during install

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 2 — Install dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Open terminal in this folder and run:

  pip install flask pandas numpy scikit-learn networkx matplotlib seaborn

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 3 — Datasets (already included!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  data/ATM_transaction_dataset.csv and data/bank_transactions_data_2.csv
  are already in this zip (SYNTHETIC demo data generated to match the
  exact column schema the agents expect) so the app runs immediately.

  ⚠️ The dataset originally uploaded with this project
  (FraudX_Combined_Dataset.csv) used a completely different column
  layout than what app.py / agents/*.py expect, so it could not be
  used as-is — that's why nothing ran. It has been removed.

  If you have your own real versions of the two CSVs (e.g. from
  Kaggle — "Bank Transaction Dataset for Fraud Detection" and an ATM
  daily-withdrawals dataset), just overwrite the two files in data/
  with the same filenames + these required columns and retrain:

  ATM_transaction_dataset.csv needs:
    atm_name, weekday, working_day (W/H), No_Of_Withdrawals,
    total_amount_withdrawn, amount_withdrawn_other_card

  bank_transactions_data_2.csv needs:
    TransactionID, AccountID, TransactionAmount, TransactionDate,
    TransactionType, Location, DeviceID, IP Address, MerchantID,
    Channel, CustomerAge, TransactionDuration, LoginAttempts,
    AccountBalance, PreviousTransactionDate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 4 — Run
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python START.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 5 — Open browser
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  http://localhost:5050

  Wait ~15 seconds for the green dot (models loading)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ❌ ModuleNotFoundError  → run pip install again
  ❌ Port in use          → change port=5050 to 5051 in app.py
  ❌ FileNotFoundError    → check your CSV files are in data/ folder

