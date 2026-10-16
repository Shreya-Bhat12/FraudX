"""
agents/bank_agent.py
──────────────────────────────────────────────────────────────────
Bank Fraud Detection Agent

Covers: Bank Transfer / UPI / NEFT / RTGS / Cheque / Branch

Detects:
  • Unusual login patterns
  • Large transfers relative to balance
  • Velocity anomalies (many fast transactions)
  • Account takeover signals
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from core.base_agent import BaseAgent


class BankFraudAgent(BaseAgent):

    FEATURE_NAMES = [
        "amount", "balance", "login_attempts", "duration_seconds",
        "customer_age", "hour", "day_of_week", "month",
        "is_weekend", "amount_zscore", "balance_ratio",
        "high_login", "fast_txn", "days_since_prev",
    ]

    def __init__(self):
        super().__init__(name="BANK_AGENT", fraud_type="BANK_FRAUD")
        self._model   = None
        self._scaler  = StandardScaler()
        self._amount_mean = 297.0
        self._amount_std  = 291.0
        self._trained     = False

    # ── training ───────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        df = df.copy()

        # Rename if raw bank CSV
        col_map = {
            "TransactionAmount":   "amount",
            "AccountBalance":      "balance",
            "LoginAttempts":       "login_attempts",
            "TransactionDuration": "duration_seconds",
            "CustomerAge":         "customer_age",
            "TransactionDate":     "timestamp",
            "PreviousTransactionDate": "prev_txn_date",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                  inplace=True)

        # Parse dates
        for col in ["timestamp", "prev_txn_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "timestamp" in df.columns:
            df["hour"]        = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["month"]       = df["timestamp"].dt.month
            df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
        else:
            df["hour"] = df["day_of_week"] = df["month"] = df["is_weekend"] = 0

        if "prev_txn_date" in df.columns and "timestamp" in df.columns:
            df["days_since_prev"] = (
                (df["timestamp"] - df["prev_txn_date"]).dt.days.abs().fillna(30)
            )
        else:
            df["days_since_prev"] = 30

        # Engineered features
        self._amount_mean = df["amount"].mean()
        self._amount_std  = df["amount"].std()
        df["amount_zscore"] = (df["amount"] - self._amount_mean) / (self._amount_std + 1e-9)
        df["balance_ratio"] = df["amount"] / (df["balance"] + 1)
        df["high_login"]    = (df["login_attempts"] > 2).astype(int)
        df["fast_txn"]      = (df["duration_seconds"] < 30).astype(int)

        if "customer_age" not in df.columns:
            df["customer_age"] = 35

        # Fraud label — a big amount ALONE is never fraud.
        # A high amount only counts when something else looks wrong too.
        other_signal = (
            (df["login_attempts"] > 1) |
            (df["duration_seconds"] < 30) |
            (df["balance_ratio"] > 0.5)
        )
        df["fraud"] = (
            (df["login_attempts"] >= 3) |
            ((df["amount_zscore"] > 2.5) & other_signal) |
            ((df["balance_ratio"] > 0.8) & (df["login_attempts"] > 1)) |
            ((df["duration_seconds"] < 20) & (df["amount"] > 500))
        ).astype(int)

        feat_cols = self.FEATURE_NAMES
        X = df[feat_cols].fillna(0).values
        y = df["fraud"].values

        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)
        self._scaler.fit(X_tr)
        Xsc = self._scaler.transform(X_tr)

        self._model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08,
            max_depth=5, random_state=42
        )
        self._model.fit(Xsc, y_tr)
        self._trained = True
        print(f"  [{self.name}] ✅ Trained on {len(df):,} rows | "
              f"Fraud: {y.sum()} ({y.mean()*100:.1f}%)")

    # ── BaseAgent implementation ───────────────────────────────
    def _extract_features(self, txn: Dict) -> np.ndarray:
        amt = float(txn.get("amount", 0))
        bal = float(txn.get("balance", 1))
        z   = (amt - self._amount_mean) / (self._amount_std + 1e-9)
        log = int(txn.get("login_attempts", 1))
        dur = float(txn.get("duration_seconds", 60))

        features = np.array([
            amt,
            bal,
            log,
            dur,
            float(txn.get("customer_age", 35)),
            float(txn.get("hour", 12)),
            float(txn.get("day_of_week", 0)),
            float(txn.get("month", 1)),
            float(txn.get("is_weekend", 0)),
            z,
            amt / (bal + 1),
            int(log > 2),
            int(dur < 30),
            float(txn.get("days_since_prev", 30)),
        ], dtype=float).reshape(1, -1)

        return self._scaler.transform(features) if self._trained else features

    def _run_model(self, features: np.ndarray) -> Tuple[float, float]:
        if not self._trained:
            return self._rule_based(features[0]), 0.5
        prob = float(self._model.predict_proba(features)[0, 1])
        conf = float(np.max(self._model.predict_proba(features)[0]))
        return prob, conf

    def _rule_based(self, f: np.ndarray) -> float:
        score = 0.05
        if f[2] >= 3:   score += 0.35   # login_attempts
        if f[9] > 2.5:  score += 0.30   # z-score
        if f[10] > 0.8: score += 0.20   # balance_ratio
        if f[3] < 20:   score += 0.10   # fast_txn
        return min(score, 0.95)

    def _explain(self, txn: Dict, features: np.ndarray, prob: float) -> Dict:
        amt = float(txn.get("amount", 0))
        bal = float(txn.get("balance", 1))
        z   = (amt - self._amount_mean) / (self._amount_std + 1e-9)
        log = int(txn.get("login_attempts", 1))
        dur = float(txn.get("duration_seconds", 60))
        return {
            "fraud_probability": round(prob, 4),
            "amount":            amt,
            "balance":           bal,
            "balance_ratio":     round(amt/(bal+1), 3),
            "amount_zscore":     round(z, 3),
            "login_attempts":    log,
            "duration_seconds":  dur,
            "risk_signals": [
                "MULTIPLE_LOGINS"    if log >= 3   else None,
                "HIGH_AMOUNT"        if z > 2.5    else None,
                "BALANCE_DRAIN"      if amt/(bal+1) > 0.8 else None,
                "FAST_TRANSACTION"   if dur < 20   else None,
            ],
        }

    def _feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def _threshold(self) -> float:
        return 0.45
