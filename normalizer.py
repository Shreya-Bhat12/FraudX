"""
core/normalizer.py
──────────────────────────────────────────────────────────────────
Layer 1 — Feature Normalization

Converts ANY incoming transaction (ATM / Bank / UPI / Credit Card /
Cheque) into a single canonical dict before it reaches the agents.
"""

import re
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional


# ── Canonical schema (every agent sees these keys) ────────────
CANONICAL_KEYS = [
    "transaction_id",   "account_id",       "amount",
    "timestamp",        "hour",             "day_of_week",
    "month",            "is_weekend",       "is_holiday",
    "location",         "device_id",        "ip_address",
    "channel",          "transaction_type", "login_attempts",
    "balance",          "duration_seconds", "merchant_id",
    "customer_age",     "days_since_prev",  "raw_channel",
]

CHANNEL_MAP = {
    # ATM variants
    "atm":        "ATM",
    "cash":       "ATM",
    # Online / Credit Card
    "online":     "ONLINE",
    "web":        "ONLINE",
    "creditcard": "ONLINE",
    "credit_card":"ONLINE",
    "cc":         "ONLINE",
    "pos":        "ONLINE",
    # Bank Transfer / UPI
    "transfer":   "TRANSFER",
    "upi":        "TRANSFER",
    "neft":       "TRANSFER",
    "rtgs":       "TRANSFER",
    "imps":       "TRANSFER",
    # Cheque
    "cheque":     "CHEQUE",
    "check":      "CHEQUE",
    # Branch
    "branch":     "BRANCH",
    "teller":     "BRANCH",
}


class FeatureNormalizer:
    """
    Normalizes raw transaction dicts into the canonical schema.
    Handles missing keys gracefully with sensible defaults.
    """

    def normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        txn = {}

        # ── IDs ────────────────────────────────────────────────
        txn["transaction_id"] = str(
            raw.get("transaction_id") or
            raw.get("TransactionID") or
            self._generate_id(raw)
        )
        txn["account_id"] = str(
            raw.get("account_id") or
            raw.get("AccountID") or
            raw.get("atm_name") or
            "UNKNOWN"
        )

        # ── Amount ─────────────────────────────────────────────
        txn["amount"] = float(
            raw.get("amount") or
            raw.get("TransactionAmount") or
            raw.get("total_amount_withdrawn") or
            0.0
        )

        # ── Timestamp ──────────────────────────────────────────
        ts = (raw.get("timestamp") or
              raw.get("TransactionDate") or
              raw.get("transaction_date") or
              datetime.now().isoformat())
        dt = self._parse_dt(ts)
        txn["timestamp"]    = dt.isoformat()
        txn["hour"]         = dt.hour
        txn["day_of_week"]  = dt.weekday()          # 0=Mon
        txn["month"]        = dt.month
        txn["is_weekend"]   = int(dt.weekday() >= 5)

        # ── Holiday flag ───────────────────────────────────────
        wday = str(raw.get("working_day", "W")).upper()
        txn["is_holiday"] = int(wday == "H")

        # ── Location / Device ──────────────────────────────────
        txn["location"]  = str(raw.get("location") or
                               raw.get("Location") or
                               raw.get("atm_name") or "UNKNOWN")
        txn["device_id"] = str(raw.get("device_id") or
                               raw.get("DeviceID") or "UNKNOWN")
        txn["ip_address"]= str(raw.get("ip_address") or
                               raw.get("IP Address") or "0.0.0.0")

        # ── Channel normalisation ──────────────────────────────
        raw_ch = str(raw.get("channel") or
                     raw.get("Channel") or "UNKNOWN").lower().strip()
        txn["raw_channel"]       = raw_ch
        txn["channel"]           = CHANNEL_MAP.get(raw_ch, raw_ch.upper())

        # ── Transaction type ───────────────────────────────────
        txn["transaction_type"] = str(
            raw.get("transaction_type") or
            raw.get("TransactionType") or
            "DEBIT"
        ).upper()

        # ── Behavioural features ───────────────────────────────
        txn["login_attempts"]  = int(raw.get("login_attempts") or
                                     raw.get("LoginAttempts") or 1)
        txn["balance"]         = float(raw.get("balance") or
                                       raw.get("AccountBalance") or 0.0)
        txn["duration_seconds"]= float(raw.get("duration_seconds") or
                                       raw.get("TransactionDuration") or 60.0)
        txn["merchant_id"]     = str(raw.get("merchant_id") or
                                     raw.get("MerchantID") or "NONE")
        txn["customer_age"]    = int(raw.get("customer_age") or
                                     raw.get("CustomerAge") or 35)
        txn["days_since_prev"] = float(raw.get("days_since_prev") or 30.0)

        return txn

    # ── helpers ────────────────────────────────────────────────
    @staticmethod
    def _parse_dt(val) -> datetime:
        if isinstance(val, datetime):
            return val
        fmts = [
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
            "%d-%m-%Y",           "%Y-%m-%d",
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(str(val)[:19], fmt)
            except ValueError:
                continue
        return datetime.now()

    @staticmethod
    def _generate_id(raw: dict) -> str:
        h = hashlib.md5(str(sorted(raw.items())).encode()).hexdigest()[:8]
        return f"AUTO_{h}"
