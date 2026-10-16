"""
core/feedback.py
──────────────────────────────────────────────────────────────────
Feedback & Learning Agent

Closes the loop of the agentic pipeline:

    Transaction → Reactive → Detection → Decision → Action → FEEDBACK
                      ▲                                         │
                      └──────────── rule updates ───────────────┘

Every time feedback arrives (the confirmed true label of a processed
transaction) this agent compares the system's decision with the truth and
rewrites the rule base owned by the REACTIVE AGENT:

    • routing / escalation rules   (HIGH_AMOUNT, HIGH_LOGIN, FAST_TXN, GRAPH_AMOUNT)
    • alert / block thresholds     (ALERT_THRESHOLD, BLOCK_THRESHOLD)
    • per-agent trust weights      (WEIGHT_<AGENT>)

Algorithm (online, rule-based learning)
───────────────────────────────────────
    outcome = compare(decision, truth)  ->  TP | TN | FP | FN

    FN  (fraud was missed)        → be MORE sensitive
          lower ALERT / BLOCK thresholds, lower the amount / login rule
          limits, widen the fast-txn window, trust agents that detected it
    FP  (legit was flagged)       → be LESS sensitive
          raise ALERT / BLOCK thresholds, raise amount / login limits,
          narrow the fast-txn window, trust agents that over-scored less
    TP / TN (correct)             → reinforce: nudge weights towards the
          agents that were right (small step) so the rule base keeps adapting

    Every change is clamped to a safe range, weights are re-normalised
    to sum to 1.0, and each change is returned so the dashboard can show
    exactly which rule changed, from what, to what, and why.
"""

import threading
import time
from core.labels import friendly, label
from typing import Dict, List, Optional

LR_THRESHOLD = 0.03      # step for alert threshold (block moves half as much)
LR_WEIGHT    = 0.03      # step for agent weights on a mistake
LR_REINFORCE = 0.005     # step for agent weights on a correct decision
REL_STEP     = 0.10      # 10% relative step for amount-style rules

BOUNDS = {
    "ALERT_THRESHOLD":        (0.30, 0.80),
    "BLOCK_THRESHOLD":        (0.55, 0.95),
    "HIGH_AMOUNT_THRESHOLD":  (150.0, 5000.0),
    "GRAPH_AMOUNT_THRESHOLD": (300.0, 10000.0),
    "HIGH_LOGIN_THRESHOLD":   (2, 6),
    "FAST_TXN_THRESHOLD":     (10, 60),
}
W_MIN, W_MAX = 0.05, 0.60

MAX_RERUNS    = 3        # how many times the loop may go back to the Reactive Agent
DOUBT_MARGIN  = 0.06     # risk this close to the ALERT/BLOCK line is 'borderline'
DISAGREE_GAP  = 0.45     # max-min agent score gap that counts as disagreement


class FeedbackAgent:
    def __init__(self, router, risk_engine):
        self.router = router
        self.risk   = risk_engine
        self._lock  = threading.RLock()
        self.version = 0
        self.history: List[Dict] = []
        self.counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    # ── rule base access (what the Reactive Agent currently uses) ──
    def snapshot(self) -> Dict[str, float]:
        snap = {}
        snap.update(self.router.get_rules())
        snap.update(self.risk.get_rules())
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in snap.items()}

    def metrics(self) -> Dict:
        c = self.counts
        tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
        total = tp + fp + fn + tn
        return {
            **c, "total": total,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "recall":    round(tp / (tp + fn), 3) if (tp + fn) else None,
            "accuracy":  round((tp + tn) / total, 3) if total else None,
        }

    # ── main entry ────────────────────────────────────────────────
    def give_feedback(self, record: Dict, true_is_fraud: bool) -> Dict:
        """
        record: stored info about the processed txn
                {txn, routing, agent_scores, risk_score, action}
        """
        with self._lock:
            before = self.snapshot()
            action = record["action"]                       # BLOCK | ALERT | ALLOW
            flagged = action in ("BLOCK", "ALERT")
            if flagged and true_is_fraud:       outcome = "TP"
            elif flagged and not true_is_fraud: outcome = "FP"
            elif not flagged and true_is_fraud: outcome = "FN"
            else:                               outcome = "TN"
            self.counts[outcome] += 1

            txn    = record["txn"]
            scores = record["agent_scores"]
            msgs: List[str] = []
            changes: List[Dict] = []

            truth_txt = "FRAUD" if true_is_fraud else "LEGITIMATE"
            msgs.append(f"Feedback received for {txn['transaction_id']}: confirmed label = {truth_txt}; "
                        f"system decision was {action} (risk {record['risk_score']:.4f}).")
            msgs.append({
                "TP": "Outcome TRUE POSITIVE — fraud correctly caught. Reinforcing agents that were right.",
                "TN": "Outcome TRUE NEGATIVE — legitimate payment correctly allowed. Reinforcing agents that were right.",
                "FP": "Outcome FALSE POSITIVE — a legitimate customer was flagged. Reactive Agent rules must become LESS strict.",
                "FN": "Outcome FALSE NEGATIVE — fraud slipped through. Reactive Agent rules must become MORE strict.",
            }[outcome])

            def setr(name, new, why, group):
                lo, hi = BOUNDS.get(name, (None, None))
                old = self._get(name)
                if lo is not None:
                    new = max(lo, min(hi, new))
                if isinstance(old, int) and not isinstance(old, bool):
                    new = int(round(new))
                else:
                    new = round(float(new), 4)
                if new == old:
                    msgs.append(f"{label(name)} is already at its safe limit ({old}) — unchanged.")
                    return
                self._set(name, new)
                changes.append({"rule": name, "label": label(name), "group": group, "old": old, "new": new,
                                "direction": "up" if new > old else "down", "why": why})

            amount = float(txn.get("amount", 0))
            logins = int(txn.get("login_attempts", 1))
            dur    = float(txn.get("duration_seconds", 60))
            r = self.router.get_rules()

            if outcome == "FN":
                setr("ALERT_THRESHOLD", self.risk.alert_threshold - LR_THRESHOLD,
                     "Fraud was allowed → lower the bar for raising an alert", "Alert threshold")
                setr("BLOCK_THRESHOLD", self.risk.block_threshold - LR_THRESHOLD / 2,
                     "Fraud was allowed → lower the bar for blocking", "Alert threshold")
                setr("HIGH_AMOUNT_THRESHOLD", r["HIGH_AMOUNT_THRESHOLD"] * (1 - REL_STEP),
                     f"Missed fraud of ₹{amount:,.0f} → cross-check smaller amounts with BANK_AGENT", "Routing rule")
                if logins < r["HIGH_LOGIN_THRESHOLD"] and logins >= 2:
                    setr("HIGH_LOGIN_THRESHOLD", r["HIGH_LOGIN_THRESHOLD"] - 1,
                         f"Fraud had {logins} login attempts (< limit) → escalate to CREDIT_AGENT sooner", "Routing rule")
                if dur > r["FAST_TXN_THRESHOLD"]:
                    setr("FAST_TXN_THRESHOLD", max(r["FAST_TXN_THRESHOLD"] + 5, min(dur, r["FAST_TXN_THRESHOLD"] + 10)),
                         f"Fraud took {dur:.0f}s (> fast limit) → widen the fast-transaction window", "Routing rule")
                if amount <= r["GRAPH_AMOUNT_THRESHOLD"]:
                    setr("GRAPH_AMOUNT_THRESHOLD", r["GRAPH_AMOUNT_THRESHOLD"] * (1 - REL_STEP),
                         "Fraud was below the graph-check amount → run GRAPH_AGENT on smaller amounts", "Routing rule")
                changes += self._shift_weights(scores, want_high=True, step=LR_WEIGHT,
                                    why_up="said fraud (≥0.5) — trust it more",
                                    why_down="said safe (<0.5) — trust it less")
            elif outcome == "FP":
                setr("ALERT_THRESHOLD", self.risk.alert_threshold + LR_THRESHOLD,
                     "Legit payment flagged → raise the bar for alerts", "Alert threshold")
                setr("BLOCK_THRESHOLD", self.risk.block_threshold + LR_THRESHOLD / 2,
                     "Legit payment flagged → raise the bar for blocking", "Alert threshold")
                if amount > r["HIGH_AMOUNT_THRESHOLD"]:
                    setr("HIGH_AMOUNT_THRESHOLD", r["HIGH_AMOUNT_THRESHOLD"] * (1 + REL_STEP),
                         f"₹{amount:,.0f} triggered extra checks but was legit → relax amount rule", "Routing rule")
                if logins >= r["HIGH_LOGIN_THRESHOLD"]:
                    setr("HIGH_LOGIN_THRESHOLD", r["HIGH_LOGIN_THRESHOLD"] + 1,
                         f"{logins} logins were legit → relax login rule", "Routing rule")
                if dur <= r["FAST_TXN_THRESHOLD"]:
                    setr("FAST_TXN_THRESHOLD", r["FAST_TXN_THRESHOLD"] - 5,
                         f"A {dur:.0f}s transaction was legit → narrow the fast-txn window", "Routing rule")
                if amount > r["GRAPH_AMOUNT_THRESHOLD"]:
                    setr("GRAPH_AMOUNT_THRESHOLD", r["GRAPH_AMOUNT_THRESHOLD"] * (1 + REL_STEP),
                         "Graph check on this amount was unnecessary → relax graph-check amount", "Routing rule")
                changes += self._shift_weights(scores, want_high=False, step=LR_WEIGHT,
                                    why_up="said safe (<0.5) — trust it more",
                                    why_down="wrongly said fraud (≥0.5) — trust it less")
            else:  # TP / TN → reinforcement only
                changes += self._shift_weights(scores, want_high=(outcome == "TP"), step=LR_REINFORCE,
                                    why_up="was right — trust it slightly more",
                                    why_down="was wrong — trust it slightly less")

            if changes:
                self.version += 1        # only a real rule rewrite creates a new rule-base version
            after = self.snapshot()
            for c in changes:
                msgs.append(f"RULE CHANGED → {c['label']}: {c['old']} → {c['new']}  ({c['why']})")
            if changes:
                msgs.append(f"Rule base updated to v{self.version}. The Reactive Agent will use these rules "
                            f"for the very next transaction.")
            elif outcome in ("TP", "TN"):
                msgs.append("Decision was CORRECT and every agent that ran pulled the same way (or only one ran), "
                            "so there is nothing to re-balance — Reactive Agent rules stay unchanged "
                            f"(v{self.version}). Rules only get rewritten on mistakes (FP / FN) or when agents disagree.")
            else:
                msgs.append(f"Mistake noted, but every affected rule is already at its safe limit — "
                            f"Reactive Agent rules unchanged (v{self.version}).")
            m = self.metrics()
            msgs.append(f"Running score: TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}"
                        + (f" | precision={m['precision']}" if m['precision'] is not None else "")
                        + (f" | recall={m['recall']}" if m['recall'] is not None else ""))

            entry = {
                "transaction_id": txn["transaction_id"], "outcome": outcome, "true_fraud": bool(true_is_fraud),
                "decision": action, "risk_score": record["risk_score"], "changes": changes,
                "rules_version": self.version, "timestamp": time.time(),
            }
            self.history.append(entry)
            self.history = self.history[-200:]
            return {
                "outcome": outcome, "true_fraud": bool(true_is_fraud), "decision": action,
                "target_agent": "REACTIVE_AGENT", "messages": msgs, "changes": changes,
                "rules_before": before, "rules_after": after,
                "rules_version": self.version, "metrics": m,
            }

    # ── autonomous loop (no human label) ──────────────────────────
    def self_review(self, decision, action_type: str, n_ran: int, n_total: int) -> Dict:
        """Is this decision trustworthy? Returns the individual doubt tests + overall verdict."""
        risk, AT, BT = decision.risk_score, self.risk.alert_threshold, self.risk.block_threshold
        sigs   = decision.risk_signals
        scores = list(decision.agent_scores.values())
        consensus = any(str(x).startswith("MULTI_AGENT_CONSENSUS") for x in sigs)
        tests = []

        near_a, near_b = abs(risk - AT) <= DOUBT_MARGIN, abs(risk - BT) <= DOUBT_MARGIN
        if consensus:
            tests.append({"name": "Borderline score", "doubt": False,
                          "detail": f"risk {risk:.3f} sits near a line, but 2+ agents independently agree → trusted"})
        else:
            line = f"ALERT line {AT:.2f}" if near_a else f"BLOCK line {BT:.2f}" if near_b else None
            tests.append({"name": "Borderline score", "doubt": bool(line),
                          "detail": (f"risk {risk:.3f} is within {DOUBT_MARGIN} of the {line}" if line
                                     else f"risk {risk:.3f} is clearly away from the ALERT ({AT:.2f}) / BLOCK ({BT:.2f}) lines")})
        gap = (max(scores) - min(scores)) if len(scores) >= 2 else 0.0
        if consensus:
            tests.append({"name": "Agent disagreement", "doubt": False,
                          "detail": f"scores range {min(scores):.2f}–{max(scores):.2f}, but 2+ agents independently flagged it → majority outweighs the outlier"})
        else:
            tests.append({"name": "Agent disagreement", "doubt": gap >= DISAGREE_GAP,
                          "detail": (f"agents disagree strongly (scores range {min(scores):.2f}–{max(scores):.2f})" if gap >= DISAGREE_GAP
                                     else f"agents roughly agree (score gap {gap:.2f})")})
        thin = action_type == "ALERT" and n_ran < n_total
        tests.append({"name": "Thin evidence", "doubt": thin,
                      "detail": (f"ALERT decided by only {n_ran} of {n_total} agents — more evidence is available" if thin
                                 else f"{n_ran} of {n_total} agents inspected it" + ("" if action_type != "ALERT" else " — full panel already used"))})
        capped = "AMOUNT_ONLY_CAPPED" in sigs
        tests.append({"name": "Amount-only suspicion", "doubt": capped,
                      "detail": ("the only red flag is a big amount — no behavioural evidence either way" if capped
                                 else "suspicion is not based on the amount alone")})
        reasons = [t["detail"] for t in tests if t["doubt"]]
        msgs = ["Self-review (no human label needed) — can this decision be trusted?"]
        msgs += [("✖ DOUBT · " if t["doubt"] else "✔ ok · ") + f"{t['name']}: {t['detail']}" for t in tests]
        return {"doubtful": bool(reasons), "reasons": reasons, "tests": tests, "messages": msgs}

    # which Reactive-Agent rule decides whether each agent joins the case
    _GATE = {"BANK_AGENT": "HIGH_AMOUNT_THRESHOLD", "CREDIT_AGENT": "HIGH_LOGIN_THRESHOLD",
             "GRAPH_AGENT": "GRAPH_AMOUNT_THRESHOLD"}
    _ORDER = ["BANK_AGENT", "CREDIT_AGENT", "GRAPH_AGENT", "ATM_AGENT"]

    def escalate(self, ran, txn: Dict, overrides: Optional[Dict]):
        """Evidence-driven rule rewrite: pick the next agent that has NOT inspected this case and rewrite
        the Reactive Agent rule that gates it, so the next pass really gathers new evidence.
        Returns (rewrites, cumulative_overrides, force_all)."""
        eff = dict(self.router.get_rules())
        eff.update(overrides or {})
        missing = [a for a in self._ORDER if a not in ran]
        agent = missing[0]
        amount, logins = float(txn.get("amount", 0)), int(txn.get("login_attempts", 1))
        ov, rewrites, force_all = dict(overrides or {}), [], False
        short = agent.replace("_AGENT", "").title()
        if agent in ("BANK_AGENT", "GRAPH_AGENT"):
            rule = self._GATE[agent]
            new = round(amount * 0.9, 2) if amount > 0 else -1.0
            why = (f"{short} check has not seen this case → big-amount limit lowered below ₹{amount:,.0f} so the Bank check joins"
                   if agent == "BANK_AGENT" else
                   f"{short} (fraud-ring) check has not seen this case → network-check amount lowered below ₹{amount:,.0f}")
        elif agent == "CREDIT_AGENT":
            rule, new = self._GATE[agent], max(1, min(int(eff[self._GATE[agent]]), logins))
            why = f"Credit check has not seen this case → login-tries limit lowered to {new} so it joins"
        else:                                    # ATM check is gated by night+fast, no single threshold admits it
            rule, new = "FULL_PANEL_CROSSCHECK", "ON"
            why = "ATM check can't be reached by one threshold (needs night+fast) → full-panel cross-check switched on"
            force_all = True
        if rule == "FULL_PANEL_CROSSCHECK":
            rewrites.append({"rule": rule, "label": label(rule), "old": "off", "new": "ON", "direction": "up", "why": why})
        else:
            old = eff[rule]
            if new != old:
                ov[rule] = new
                rewrites.append({"rule": rule, "label": label(rule), "old": old, "new": new,
                                 "direction": "up" if new > old else "down", "why": why})
        return rewrites, ov, force_all, agent

    # ── helpers ───────────────────────────────────────────────────
    def _shift_weights(self, scores, want_high: bool, step: float, why_up: str, why_down: str) -> List[Dict]:
        """Move trust toward agents that were right; renormalise to sum 1."""
        out: List[Dict] = []
        if not scores:
            return out
        w = dict(self.risk.weights)
        ran = [ag for ag in scores if ag in w]           # only agents that actually ran are re-weighted
        if not ran:
            return out
        pool = sum(w[ag] for ag in ran)                  # their combined weight stays constant
        new = {}
        for ag in ran:
            p = scores[ag]
            right = ((p >= 0.5) == want_high)
            new[ag] = max(W_MIN, min(W_MAX, w[ag] + (step if right else -step)))
        total = sum(new.values())
        new = {k: v / total * pool for k, v in new.items()}
        for ag, nv in new.items():
            ov = w[ag]
            nv = round(nv, 4)
            if abs(nv - ov) < 1e-4:
                continue
            p = scores[ag]
            right = ((p >= 0.5) == want_high)
            reason = f"{ag.replace('_AGENT','').title()} check {why_up if right else why_down} — it gave {p:.2f}"
            self.risk.weights[ag] = nv
            out.append({"rule": f"WEIGHT_{ag}", "label": label(f"WEIGHT_{ag}"), "group": "Agent weight", "old": round(ov, 4), "new": nv,
                        "direction": "up" if nv > ov else "down", "why": reason})
        return out

    def _get(self, name):
        if name == "ALERT_THRESHOLD": return round(self.risk.alert_threshold, 4)
        if name == "BLOCK_THRESHOLD": return round(self.risk.block_threshold, 4)
        return getattr(self.router, name)

    def _set(self, name, val):
        if name == "ALERT_THRESHOLD":
            self.risk.alert_threshold = val
            self.risk.block_threshold = max(self.risk.block_threshold, val + 0.10)
        elif name == "BLOCK_THRESHOLD":
            self.risk.block_threshold = max(val, self.risk.alert_threshold + 0.10)
        else:
            setattr(self.router, name, val)
