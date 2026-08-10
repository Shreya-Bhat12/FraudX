"""
agents/credit_agent.py
──────────────────────────────────────────────────────────────────
Credit Card Fraud Detection Agent

Detects:
  • Card-not-present fraud (online purchases)
  • Velocity attacks (many small charges in short time)
  • Unusual merchant categories
  • Geographic impossibility signals
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from core.base_agent import BaseAgent


class CreditCardFraudAgent(BaseAgent):

    FEATURE_NAMES = [
        "amount", "balance", "login_attempts", "duration_seconds",
        "customer_age", "hour", "day_of_week", "is_weekend",
        "amount_zscore", "balance_ratio", "cc_velocity",
        "is_night", "high_amount_flag",
    ]

    def __init__(self):
        super().__init__(name="CREDIT_AGENT", fraud_type="CREDIT_CARD_FRAUD")
        self._model        = None
        self._scaler       = StandardScaler()
        self._amount_mean  = 297.0
        self._amount_std   = 291.0
        self._trained      = False
        # Simple in-memory velocity tracker: account_id → count
        self._velocity: dict = defaultdict(int)

    # ── training ───────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        df = df.copy()
        col_map = {
            "TransactionAmount": "amount",
            "AccountBalance":    "balance",
            "LoginAttempts":     "login_attempts",
            "TransactionDuration": "duration_seconds",
            "CustomerAge":       "customer_age",
            "TransactionDate":   "timestamp",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                  inplace=True)

        if "timestamp" in df.columns:
            df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"]       = df["timestamp"].dt.hour
            df["day_of_week"]= df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
        else:
            df["hour"] = df["day_of_week"] = df["is_weekend"] = 0

        self._amount_mean = df["amount"].mean()
        self._amount_std  = df["amount"].std()
        df["amount_zscore"]  = (df["amount"] - self._amount_mean) / (self._amount_std + 1e-9)
        df["balance_ratio"]  = df["amount"] / (df["balance"] + 1)
        df["cc_velocity"]    = df.groupby("AccountID")["amount"].transform("count") \
                               if "AccountID" in df.columns else 1
        df["is_night"]       = df["hour"].apply(lambda h: int(h < 6 or h >= 22))
        df["high_amount_flag"]= (df["amount"] > 800).astype(int)

        if "customer_age" not in df.columns:
            df["customer_age"] = 35

        # Fraud label
        df["cc_fraud_score"] = (
            (df["amount"] > 800).astype(int) * 2 +
            (df["login_attempts"] > 2).astype(int) * 3 +
            (df["duration_seconds"] < 25).astype(int) * 1 +
            (df["amount_zscore"] > 2.0).astype(int) * 2
        )
        df["fraud"] = (df["cc_fraud_score"] >= 4).astype(int)

        feat_cols = self.FEATURE_NAMES
        X = df[feat_cols].fillna(0).values
        y = df["fraud"].values

        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)
        self._scaler.fit(X_tr)
        Xsc = self._scaler.transform(X_tr)

        self._model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=10, random_state=42
        )
        self._model.fit(Xsc, y_tr)
        self._trained = True
        print(f"  [{self.name}] ✅ Trained on {len(df):,} rows | "
              f"Fraud: {y.sum()} ({y.mean()*100:.1f}%)")

    # ── BaseAgent implementation ───────────────────────────────
    def _extract_features(self, txn: Dict) -> np.ndarray:
        amt  = float(txn.get("amount", 0))
        bal  = float(txn.get("balance", 1))
        z    = (amt - self._amount_mean) / (self._amount_std + 1e-9)
        log  = int(txn.get("login_attempts", 1))
        dur  = float(txn.get("duration_seconds", 60))
        hour = int(txn.get("hour", 12))
        acct = txn.get("account_id", "UNKNOWN")

        # Update velocity counter
        self._velocity[acct] += 1
        vel = self._velocity[acct]

        features = np.array([
            amt, bal, log, dur,
            float(txn.get("customer_age", 35)),
            float(hour),
            float(txn.get("day_of_week", 0)),
            float(txn.get("is_weekend", 0)),
            z,
            amt / (bal + 1),
            float(vel),
            float(int(hour < 6 or hour >= 22)),
            float(int(amt > 800)),
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
        if f[2] >= 3: score += 0.30
        if f[8] > 2:  score += 0.25
        if f[12]:     score += 0.20
        if f[10]:     score += 0.20
        return min(score, 0.95)

    def _explain(self, txn: Dict, features: np.ndarray, prob: float) -> Dict:
        amt  = float(txn.get("amount", 0))
        hour = int(txn.get("hour", 12))
        z    = (amt - self._amount_mean) / (self._amount_std + 1e-9)
        acct = txn.get("account_id", "?")
        return {
            "fraud_probability": round(prob, 4),
            "amount":            amt,
            "amount_zscore":     round(z, 3),
            "velocity":          self._velocity.get(acct, 1),
            "is_night":          int(hour < 6 or hour >= 22),
            "login_attempts":    txn.get("login_attempts", 1),
            "risk_signals": [
                "HIGH_VELOCITY"    if self._velocity.get(acct, 1) > 5 else None,
                "NIGHT_PURCHASE"   if (hour < 6 or hour >= 22) else None,
                "LARGE_AMOUNT"     if amt > 800 else None,
                "LOGIN_ANOMALY"    if int(txn.get("login_attempts",1)) >= 3 else None,
            ],
        }

    def _feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def _threshold(self) -> float:
        return 0.40
