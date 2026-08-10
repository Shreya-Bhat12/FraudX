"""
core/router.py
──────────────────────────────────────────────────────────────────
Layer 2 — Routing Agent

Classifies each normalised transaction and decides which fraud
detection agent(s) should handle it.

Rule-based routing now; pluggable ML routing later.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    primary_agent:    str          # e.g. "ATM_AGENT"
    secondary_agents: List[str]    # additional agents to also run
    fraud_type:       str          # ATM / CREDIT_CARD / UPI / CHEQUE / BANK
    confidence:       float        # routing confidence 0–1
    reason:           str


class RoutingAgent:
    """
    Decides which agent(s) should inspect a transaction.

    Rules (in priority order):
        1. Channel-based primary routing
        2. Amount/behaviour overrides → escalate to multiple agents
        3. High-risk indicators → always include GRAPH agent
    """

    # channel → (primary_agent, fraud_type)
    CHANNEL_ROUTES: Dict[str, Tuple[str, str]] = {
        "ATM":      ("ATM_AGENT",    "ATM_FRAUD"),
        "ONLINE":   ("CREDIT_AGENT", "CREDIT_CARD_FRAUD"),
        "TRANSFER": ("BANK_AGENT",   "UPI_BANK_FRAUD"),
        "CHEQUE":   ("BANK_AGENT",   "CHEQUE_FRAUD"),
        "BRANCH":   ("BANK_AGENT",   "BANK_FRAUD"),
    }

    # Thresholds that trigger additional agents
    HIGH_AMOUNT_THRESHOLD  = 500.0
    HIGH_LOGIN_THRESHOLD   = 3
    FAST_TXN_THRESHOLD     = 20   # seconds
    SUSPICIOUS_HOUR_START  = 0
    SUSPICIOUS_HOUR_END    = 5

    def route(self, txn: Dict) -> RoutingDecision:
        channel = txn.get("channel", "UNKNOWN").upper()
        amount  = float(txn.get("amount", 0))
        logins  = int(txn.get("login_attempts", 1))
        dur     = float(txn.get("duration_seconds", 60))
        hour    = int(txn.get("hour", 12))

        # ── Primary routing ────────────────────────────────────
        if channel in self.CHANNEL_ROUTES:
            primary, fraud_type = self.CHANNEL_ROUTES[channel]
            confidence = 0.95
            reason = f"Channel={channel} → {primary}"
        else:
            # Fallback: amount-based
            primary    = "BANK_AGENT"
            fraud_type = "UNKNOWN_FRAUD"
            confidence = 0.60
            reason     = f"Unknown channel '{channel}', defaulting to BANK_AGENT"

        # ── Secondary escalation rules ─────────────────────────
        secondary = []

        # Large amount always cross-checks with bank agent
        if amount > self.HIGH_AMOUNT_THRESHOLD and primary != "BANK_AGENT":
            secondary.append("BANK_AGENT")
            reason += f" | High amount={amount:.0f} → +BANK_AGENT"

        # Multiple logins → credit-card style attack pattern
        if logins >= self.HIGH_LOGIN_THRESHOLD and primary != "CREDIT_AGENT":
            secondary.append("CREDIT_AGENT")
            reason += f" | High logins={logins} → +CREDIT_AGENT"

        # Night-time + fast transaction → ATM skimming pattern
        is_night  = hour <= self.SUSPICIOUS_HOUR_END or hour >= 23
        is_fast   = dur  <= self.FAST_TXN_THRESHOLD
        if is_night and is_fast and primary != "ATM_AGENT":
            secondary.append("ATM_AGENT")
            reason += f" | Night+fast → +ATM_AGENT"

        # High-risk profile → always run graph check
        if amount > 1000 or logins >= 4 or (is_night and is_fast):
            secondary.append("GRAPH_AGENT")
            reason += " | High-risk → +GRAPH_AGENT"

        # Remove duplicates while preserving order
        secondary = list(dict.fromkeys(
            [a for a in secondary if a != primary]
        ))

        return RoutingDecision(
            primary_agent    = primary,
            secondary_agents = secondary,
            fraud_type       = fraud_type,
            confidence       = confidence,
            reason           = reason,
        )

    def describe(self, decision: RoutingDecision) -> str:
        agents = [decision.primary_agent] + decision.secondary_agents
        return (
            f"  🔀 Router → Primary: {decision.primary_agent} "
            f"| Secondary: {decision.secondary_agents or 'none'}\n"
            f"     FraudType: {decision.fraud_type} "
            f"| Reason: {decision.reason}"
        )
