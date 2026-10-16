"""core/labels.py — plain-English names for every rule shown in the dashboard."""
import re

LABELS = {
    "HIGH_AMOUNT_THRESHOLD":  ("Big-amount limit (₹)",
        "If a payment is above this amount, the Bank check is also run. Lower = stricter."),
    "HIGH_LOGIN_THRESHOLD":   ("Login-tries limit",
        "This many wrong/repeated logins makes the Credit check run too. Lower = stricter."),
    "FAST_TXN_THRESHOLD":     ("'Too fast' limit (seconds)",
        "A late-night payment finished faster than this looks like a bot or skimmer, so the ATM check also runs."),
    "GRAPH_AMOUNT_THRESHOLD": ("Network-check amount (₹)",
        "Above this amount we also look at who the account sends money to (fraud-ring check)."),
    "FULL_PANEL_CROSSCHECK":  ("Full-panel cross-check",
        "Last-resort escalation from the Feedback Agent: every agent inspects the transaction."),
    "ALERT_THRESHOLD":        ("Risk score that raises an ALERT",
        "Risk score at or above this → customer is asked to verify (OTP). Lower = more alerts."),
    "BLOCK_THRESHOLD":        ("Risk score that BLOCKS",
        "Risk score at or above this → payment is declined. Lower = more blocks."),
    "WEIGHT_BANK_AGENT":      ("Trust in Bank check",   "Share of the final risk score that comes from the Bank check."),
    "WEIGHT_CREDIT_AGENT":    ("Trust in Card check",   "Share of the final risk score that comes from the Card check."),
    "WEIGHT_ATM_AGENT":       ("Trust in ATM check",    "Share of the final risk score that comes from the ATM check."),
    "WEIGHT_GRAPH_AGENT":     ("Trust in Network check","Share of the final risk score that comes from the Network (fraud-ring) check."),
}

def label(rule: str) -> str:
    return LABELS.get(rule, (rule, ""))[0]

def desc(rule: str) -> str:
    return LABELS.get(rule, (rule, ""))[1]

_PAT = re.compile("|".join(sorted(map(re.escape, LABELS), key=len, reverse=True)))

def friendly(text: str) -> str:
    """Replace raw rule names inside a sentence with their plain-English label."""
    return _PAT.sub(lambda m: label(m.group(0)), text)

def all_labels() -> dict:
    return {k: {"label": v[0], "desc": v[1]} for k, v in LABELS.items()}
