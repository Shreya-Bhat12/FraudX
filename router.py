"""
core/router.py
──────────────────────────────────────────────────────────────────
Layer 2 — Routing Agent

Classifies each normalised transaction and decides which fraud
detection agent(s) should handle it.

Rule-based routing now; pluggable ML routing later.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class RoutingDecision:
    primary_agent:    str          # e.g. "ATM_AGENT"
    secondary_agents: List[str]    # additional agents to also run
    fraud_type:       str          # ATM / CREDIT_CARD / UPI / CHEQUE / BANK
    confidence:       float        # routing confidence 0–1
    reason:           str
    rule_checks:      List[Dict] = field(default_factory=list)   # every rule evaluated (fired or not)


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

    def get_rules(self) -> Dict[str, float]:
        """Current (feedback-tunable) rule thresholds."""
        return {
            "HIGH_AMOUNT_THRESHOLD": self.HIGH_AMOUNT_THRESHOLD,
            "HIGH_LOGIN_THRESHOLD":  self.HIGH_LOGIN_THRESHOLD,
            "FAST_TXN_THRESHOLD":    self.FAST_TXN_THRESHOLD,
            "GRAPH_AMOUNT_THRESHOLD": getattr(self, "GRAPH_AMOUNT_THRESHOLD", 1000.0),
        }

    GRAPH_AMOUNT_THRESHOLD = 1000.0   # amount above which the graph agent is always added

    def route(self, txn: Dict, overrides: Dict = None, force_all: bool = False) -> RoutingDecision:
        """overrides / force_all = case-level rule rewrites sent by the Feedback Agent during a re-run."""
        ov = overrides or {}
        HIGH_AMOUNT  = ov.get("HIGH_AMOUNT_THRESHOLD",  self.HIGH_AMOUNT_THRESHOLD)
        HIGH_LOGIN   = ov.get("HIGH_LOGIN_THRESHOLD",   self.HIGH_LOGIN_THRESHOLD)
        FAST_TXN     = ov.get("FAST_TXN_THRESHOLD",     self.FAST_TXN_THRESHOLD)
        GRAPH_AMOUNT = ov.get("GRAPH_AMOUNT_THRESHOLD", self.GRAPH_AMOUNT_THRESHOLD)
        checks: List[Dict] = []
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
        hit = amount > HIGH_AMOUNT and primary != "BANK_AGENT"
        checks.append({"rule": "HIGH_AMOUNT_THRESHOLD", "test": f"amount {amount:.0f} > {HIGH_AMOUNT:.0f}",
                       "fired": bool(hit), "effect": "+BANK_AGENT"})
        if hit:
            secondary.append("BANK_AGENT")
            reason += f" | High amount={amount:.0f} → +BANK_AGENT"

        # Multiple logins → credit-card style attack pattern
        hit = logins >= HIGH_LOGIN and primary != "CREDIT_AGENT"
        checks.append({"rule": "HIGH_LOGIN_THRESHOLD", "test": f"login attempts {logins} >= {HIGH_LOGIN}",
                       "fired": bool(hit), "effect": "+CREDIT_AGENT"})
        if hit:
            secondary.append("CREDIT_AGENT")
            reason += f" | High logins={logins} → +CREDIT_AGENT"

        # Night-time + fast transaction → ATM skimming pattern
        is_night  = hour <= self.SUSPICIOUS_HOUR_END or hour >= 23
        is_fast   = dur  <= FAST_TXN
        checks.append({"rule": "FAST_TXN_THRESHOLD", "test": f"night(hour={hour})={is_night} and duration {dur:.0f}s <= {FAST_TXN:.0f}s",
                       "fired": bool(is_night and is_fast and primary != "ATM_AGENT"), "effect": "+ATM_AGENT"})
        if is_night and is_fast and primary != "ATM_AGENT":
            secondary.append("ATM_AGENT")
            reason += f" | Night+fast → +ATM_AGENT"

        # High-risk profile → always run graph check
        g_hit = amount > GRAPH_AMOUNT or logins >= 4 or (is_night and is_fast)
        checks.append({"rule": "GRAPH_AMOUNT_THRESHOLD", "test": f"amount {amount:.0f} > {GRAPH_AMOUNT:.0f} or logins>=4 or night+fast",
                       "fired": bool(g_hit), "effect": "+GRAPH_AGENT"})
        if g_hit:
            secondary.append("GRAPH_AGENT")
            reason += " | High-risk → +GRAPH_AGENT"

        # Remove duplicates while preserving order
        secondary = list(dict.fromkeys(
            [a for a in secondary if a != primary]
        ))

        if force_all:
            extra = [a for a in ("ATM_AGENT", "BANK_AGENT", "CREDIT_AGENT", "GRAPH_AGENT")
                     if a != primary and a not in secondary]
            checks.append({"rule": "FULL_PANEL_CROSSCHECK", "test": "Feedback Agent escalation: run every agent",
                           "fired": True, "effect": "+ALL_AGENTS"})
            secondary += extra
            reason += " | Feedback escalation → full panel"

        return RoutingDecision(
            primary_agent    = primary,
            secondary_agents = secondary,
            fraud_type       = fraud_type,
            confidence       = confidence,
            reason           = reason,
            rule_checks      = checks,
        )

    def describe(self, decision: RoutingDecision) -> str:
        agents = [decision.primary_agent] + decision.secondary_agents
        return (
            f"  🔀 Router → Primary: {decision.primary_agent} "
            f"| Secondary: {decision.secondary_agents or 'none'}\n"
            f"     FraudType: {decision.fraud_type} "
            f"| Reason: {decision.reason}"
        )
