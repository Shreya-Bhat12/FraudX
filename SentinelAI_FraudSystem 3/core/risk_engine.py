"""
core/risk_engine.py
──────────────────────────────────────────────────────────────────
Risk Aggregation Engine

Combines scores from multiple fraud agents into a single
weighted risk score, then makes a final Block / Alert / Allow
decision with full explainability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from core.base_agent import AgentResult


# ── Risk thresholds ────────────────────────────────────────────
BLOCK_THRESHOLD = 0.80
ALERT_THRESHOLD = 0.55
ALLOW_THRESHOLD = 0.55   # below this → allow

# ── Agent weights (must sum to 1.0) ───────────────────────────
DEFAULT_WEIGHTS = {
    "BANK_AGENT":   0.35,
    "CREDIT_AGENT": 0.30,
    "ATM_AGENT":    0.20,
    "GRAPH_AGENT":  0.15,
}


@dataclass
class RiskDecision:
    transaction_id:   str
    risk_score:       float          # 0.0 – 1.0
    action:           str            # BLOCK | ALERT | ALLOW
    primary_agent:    str            # which agent drove the decision
    agent_scores:     Dict[str, float]
    agent_flags:      Dict[str, bool]
    risk_signals:     List[str]      # human-readable reasons
    recommended_action: str
    confidence:       float


class RiskAggregationEngine:
    """
    Weighted combination of agent scores + rule overrides.

    Weighting logic:
        1. Collect fraud_probability from each AgentResult
        2. Apply weights → weighted_sum
        3. Apply override rules (e.g. graph flag → always alert)
        4. Classify: BLOCK / ALERT / ALLOW
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def aggregate(
        self,
        transaction_id: str,
        results: List[AgentResult],
    ) -> RiskDecision:

        agent_scores: Dict[str, float] = {}
        agent_flags:  Dict[str, bool]  = {}
        all_signals:  List[str]        = []
        confidences:  List[float]      = []

        for r in results:
            agent_scores[r.agent_name] = r.fraud_probability
            agent_flags[r.agent_name]  = r.fraud_flag
            confidences.append(r.confidence)

            # Collect non-null risk signals from each agent
            sigs = r.reasoning.get("risk_signals", [])
            for s in sigs:
                if s and s not in all_signals:
                    all_signals.append(s)

        # ── Weighted score ─────────────────────────────────────
        weighted_sum = 0.0
        weight_total = 0.0
        for agent_name, prob in agent_scores.items():
            w = self.weights.get(agent_name, 0.10)
            weighted_sum  += w * prob
            weight_total  += w

        # Normalise in case not all agents ran
        risk_score = weighted_sum / (weight_total + 1e-9)

        # ── Override rules ─────────────────────────────────────
        # Rule 1: If graph agent flagged → always at least ALERT
        if agent_flags.get("GRAPH_AGENT", False):
            risk_score = max(risk_score, ALERT_THRESHOLD + 0.05)
            all_signals.append("GRAPH_NETWORK_FRAUD_FLAG")

        # Rule 2: If 2+ agents all flagged → escalate to BLOCK
        n_flagged = sum(1 for f in agent_flags.values() if f)
        if n_flagged >= 2:
            risk_score = max(risk_score, BLOCK_THRESHOLD + 0.01)
            all_signals.append(f"MULTI_AGENT_CONSENSUS ({n_flagged} agents)")

        # Rule 3: Any single agent > 0.90 → at least ALERT
        if any(p > 0.90 for p in agent_scores.values()):
            risk_score = max(risk_score, ALERT_THRESHOLD + 0.02)
            all_signals.append("HIGH_CONFIDENCE_SINGLE_AGENT")

        risk_score = round(min(risk_score, 1.0), 4)

        # ── Decision ───────────────────────────────────────────
        if risk_score >= BLOCK_THRESHOLD:
            action = "🚫 BLOCK"
            recommended = "Decline transaction. Freeze account. Notify fraud team."
        elif risk_score >= ALERT_THRESHOLD:
            action = "⚠️  ALERT"
            recommended = "Send OTP / call verification to customer. Flag for review."
        else:
            action = "✅ ALLOW"
            recommended = "Transaction approved. Continue monitoring."

        # Primary driver = highest-scoring agent
        primary = max(agent_scores, key=agent_scores.get) if agent_scores else "NONE"
        avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.5

        return RiskDecision(
            transaction_id    = transaction_id,
            risk_score        = risk_score,
            action            = action,
            primary_agent     = primary,
            agent_scores      = {k: round(v, 4) for k, v in agent_scores.items()},
            agent_flags       = agent_flags,
            risk_signals      = [s for s in all_signals if s],
            recommended_action= recommended,
            confidence        = avg_conf,
        )

    def print_decision(self, d: RiskDecision):
        print(f"\n  {'─'*55}")
        print(f"  🏦  Transaction : {d.transaction_id}")
        print(f"  📊  Risk Score  : {d.risk_score:.4f}")
        print(f"  {d.action}")
        print(f"  🔍  Primary     : {d.primary_agent}")
        print(f"  📡  Agent Scores:")
        for agent, score in d.agent_scores.items():
            flag = "🚨" if d.agent_flags.get(agent) else "✅"
            print(f"        {flag} {agent:<18} {score:.4f}")
        if d.risk_signals:
            print(f"  ⚡  Signals     : {', '.join(d.risk_signals)}")
        print(f"  💡  Action      : {d.recommended_action}")
        print(f"  {'─'*55}")
