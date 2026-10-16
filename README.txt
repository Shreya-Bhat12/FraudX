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


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHAT'S NEW — AGENT DASHBOARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dashboard → "Start Detecting":
  • After EVERY agent finishes, a pop-up explains exactly what it did
    (normalising, rules checked, models run, weighted-score maths,
    "thinking" BLOCK vs ALERT vs ALLOW, action taken). Tick
    "Auto-continue" to skip clicking Continue.
  • "Agent Activity Log" keeps every message for the teacher to read.
  • Feedback Agent (core/feedback.py): the confirmed label is compared
    with the decision (TP/TN/FP/FN) and the Reactive Agent's rules are
    REWRITTEN — routing limits, alert/block thresholds, agent weights.
    The arrow from the Feedback node points back to the Reactive Agent,
    and the "Reactive Agent · Live Rule Base" panel shows the new values.
  • API: POST /api/feedback {transaction_id, true_label}, GET /api/rules


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHAT'S NEW — PREDICT ↔ DASHBOARD + AGENTIC FEEDBACK LOOP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Predict tab → "Run Analysis" → pop-up "watch it on the Dashboard".
    Open the Dashboard and it replays exactly what every agent did for
    YOUR transaction. Open the Dashboard without predicting: a random
    simulated transaction runs by itself (first visit only).
  • No human feedback. The Feedback Agent reviews its own decision
    (POST /api/predict_loop). It doubts a decision when:
      – the risk score is borderline (within 0.06 of ALERT/BLOCK line)
      – agents strongly disagree (score gap >= 0.45, no 2-agent consensus)
      – ALERT was decided on thin evidence (not all agents ran)
      – the only red flag is a big amount
  • If doubtful: Feedback Agent REWRITES the Reactive Agent's rule that
    gates the next missing agent (big-amount / login / network limit, or
    full-panel cross-check for the ATM agent) and the pipeline STARTS OVER
    (Reactive → Detection → Decision → Feedback). Each re-run adds at
    least one new agent, so it gathers new evidence; max 4 passes.
    These rewrites are scoped to that one case — the global rule base
    only learns from simulator-labelled runs (Start Detecting).
  • Alert & Action runs once, after the loop settles on the final result.
