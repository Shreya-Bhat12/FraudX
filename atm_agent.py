"""
agents/atm_agent.py
──────────────────────────────────────────────────────────────────
ATM Fraud Detection Agent

Detects:
  • Abnormal withdrawal volumes at an ATM
  • Suspicious card mix (high other-bank card ratio)
  • Holiday/weekend spikes
  • Z-score anomalies on amount & count
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from core.base_agent import BaseAgent, AgentResult


class ATMFraudAgent(BaseAgent):

    FEATURE_NAMES = [
        "amount",
        "is_weekend",
        "is_holiday",
        "day_of_week",
        "month",
        "hour",
        "login_attempts",
        "duration_seconds",
        "amount_zscore",        # computed at predict time
        "is_night",             # computed at predict time
        "is_fast_txn",          # computed at predict time
        "balance_ratio",        # computed at predict time
    ]

    def __init__(self):
        super().__init__(name="ATM_AGENT", fraud_type="ATM_FRAUD")
        self._model   = None
        self._scaler  = StandardScaler()
        self._amount_stats = {"mean": 500_000, "std": 300_000}   # defaults
        self._trained = False
        self._threshold_val = 0.45

    # ── training ───────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        """Train on ATM daily-aggregate data."""
        df = df.copy()

        # Derived features
        df["amount_zscore"] = (
            (df["total_amount_withdrawn"] - df["total_amount_withdrawn"].mean())
            / (df["total_amount_withdrawn"].std() + 1e-9)
        )
        df["other_ratio"] = (
            df["amount_withdrawn_other_card"] /
            (df["total_amount_withdrawn"] + 1)
        )
        df["is_weekend"]  = df["weekday"].str.lower().isin(["saturday","sunday"]).astype(int)
        df["is_holiday"]  = (df["working_day"] == "H").astype(int)
        df["is_night"]    = 0   # daily data — no hour info
        df["is_fast_txn"] = 0

        # Synthetic fraud label
        df["fraud"] = (
            (df["amount_zscore"] > 2.0) &
            (df["is_holiday"] == 1) &
            (df["other_ratio"] > 0.7)
        ).astype(int)
        extra = df[(df["amount_zscore"]>1.5)].sample(frac=0.015, random_state=42).index
        df.loc[extra, "fraud"] = 1

        # Store stats for z-score at inference
        self._amount_stats = {
            "mean": df["total_amount_withdrawn"].mean(),
            "std":  df["total_amount_withdrawn"].std(),
        }

        feature_cols = [
            "total_amount_withdrawn","is_weekend","is_holiday",
            "amount_zscore","other_ratio","No_Of_Withdrawals",
        ]
        X = df[feature_cols].fillna(0).values
        y = df["fraud"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self._scaler.fit(X_tr)
        X_tr_sc = self._scaler.transform(X_tr)

        self._model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=8, random_state=42
        )
        self._model.fit(X_tr_sc, y_tr)
        self._trained = True
        self._train_feature_cols = feature_cols
        print(f"  [{self.name}] ✅ Trained on {len(df):,} rows | "
              f"Fraud: {y.sum()} ({y.mean()*100:.1f}%)")

    # ── BaseAgent implementation ───────────────────────────────
    def _extract_features(self, txn: Dict) -> np.ndarray:
        amt = float(txn.get("amount", 0))
        z   = (amt - self._amount_stats["mean"]) / (self._amount_stats["std"] + 1e-9)
        bal = float(txn.get("balance", 1))
        features = np.array([
            amt,
            int(txn.get("is_weekend", 0)),
            int(txn.get("is_holiday", 0)),
            z,
            amt / (bal + 1),         # balance_ratio proxy
            int(txn.get("login_attempts", 1)),
        ], dtype=float).reshape(1, -1)
        return self._scaler.transform(features) if self._trained else features

    def _run_model(self, features: np.ndarray) -> Tuple[float, float]:
        if not self._trained or self._model is None:
            return self._rule_based(features), 0.5
        prob  = self._model.predict_proba(features)[0, 1]
        probs = self._model.predict_proba(features)[0]
        conf  = float(np.max(probs))
        return float(prob), conf

    def _rule_based(self, features: np.ndarray) -> float:
        """Fallback when model not trained."""
        amt_z = float(features[0, 3]) if features.shape[1] > 3 else 0
        score = 0.1
        if amt_z > 2.5: score += 0.4
        if amt_z > 1.5: score += 0.2
        return min(score, 0.95)

    def _explain(self, txn: Dict, features: np.ndarray, prob: float) -> Dict:
        amt = float(txn.get("amount", 0))
        z   = (amt - self._amount_stats["mean"]) / (self._amount_stats["std"] + 1e-9)
        return {
            "fraud_probability": round(prob, 4),
            "amount":            amt,
            "amount_zscore":     round(z, 3),
            "is_holiday":        txn.get("is_holiday", 0),
            "is_weekend":        txn.get("is_weekend", 0),
            "risk_signals": [
                "HIGH_AMOUNT_SPIKE"    if z > 2.0 else None,
                "HOLIDAY_TRANSACTION"  if txn.get("is_holiday") else None,
                "WEEKEND_TRANSACTION"  if txn.get("is_weekend") else None,
            ],
        }

    def _feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def _threshold(self) -> float:
        return self._threshold_val
