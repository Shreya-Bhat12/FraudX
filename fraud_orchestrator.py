"""
fraud_orchestrator.py — Master Agentic Controller
"""
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import time
import pandas as pd
import numpy as np
from typing import Dict, List

from core.normalizer  import FeatureNormalizer
from core.router      import RoutingAgent
from core.risk_engine import RiskAggregationEngine
from core.base_agent  import AgentResult
from core.feedback    import FeedbackAgent
from core.labels      import friendly, label
from agents.atm_agent    import ATMFraudAgent
from agents.bank_agent   import BankFraudAgent
from agents.credit_agent import CreditCardFraudAgent
from agents.graph_agent  import GraphFraudAgent


class FraudOrchestrator:
    def __init__(self):
        print("🚀 Initialising Fraud Orchestrator...")
        self.normalizer  = FeatureNormalizer()
        self.router      = RoutingAgent()
        self.risk_engine = RiskAggregationEngine()
        self.agents = {
            "ATM_AGENT":    ATMFraudAgent(),
            "BANK_AGENT":   BankFraudAgent(),
            "CREDIT_AGENT": CreditCardFraudAgent(),
            "GRAPH_AGENT":  GraphFraudAgent(),
        }
        self.feedback    = FeedbackAgent(self.router, self.risk_engine)
        self._records: Dict[str, Dict] = {}     # processed txns awaiting feedback
        self._trained   = False
        self._audit_log: List[Dict] = []

    def train_all(self, atm_path: str, bank_path: str):
        print("\n📚 Training all agents on historical data...")
        atm_df  = pd.read_csv(atm_path)
        bank_df = pd.read_csv(bank_path)

        self.agents["ATM_AGENT"].train(atm_df)
        self.agents["BANK_AGENT"].train(bank_df)

        cc_df = bank_df[bank_df["Channel"].isin(["Online","ATM"])].copy() \
                if "Channel" in bank_df.columns else bank_df.copy()
        self.agents["CREDIT_AGENT"].train(cc_df)

        self.agents["GRAPH_AGENT"].build_from_dataframe(
            bank_df, src_col="AccountID", dst_col="MerchantID",
            amt_col="TransactionAmount",
        )
        self._trained = True
        print("\n✅ All agents trained and ready.\n")

    def process(self, raw_transaction: Dict, verbose: bool = True) -> Dict:
        t0  = time.time()
        txn = self.normalizer.normalize(raw_transaction)
        txn_id = txn["transaction_id"]

        if verbose:
            print(f"\n{'═'*58}")
            print(f"  📥 Transaction: {txn_id}")
            print(f"     Account : {txn['account_id']}")
            print(f"     Amount  : ₹{txn['amount']:,.2f}")
            print(f"     Channel : {txn['channel']}  ({txn['transaction_type']})")

        routing = self.router.route(txn)
        if verbose:
            print(self.router.describe(routing))

        results: List[AgentResult] = []
        for agent_name in [routing.primary_agent] + routing.secondary_agents:
            agent = self.agents.get(agent_name)
            if agent:
                result = agent.predict(txn)
                results.append(result)
                if verbose:
                    flag = "🚨" if result.fraud_flag else "✅"
                    print(f"     {flag} {agent_name:<18} prob={result.fraud_probability:.4f}  conf={result.confidence:.2f}  ({result.latency_ms:.1f}ms)")

        decision = self.risk_engine.aggregate(txn_id, results)
        if verbose:
            self.risk_engine.print_decision(decision)

        total_ms = round((time.time() - t0) * 1000, 1)
        self._audit_log.append({
            "transaction_id": txn_id,
            "amount":         txn["amount"],
            "channel":        txn["channel"],
            "risk_score":     decision.risk_score,
            "action":         decision.action,
            "agents_run":     [routing.primary_agent] + routing.secondary_agents,
            "agent_scores":   decision.agent_scores,
            "signals":        decision.risk_signals,
            "total_ms":       total_ms,
        })
        action_type = "BLOCK" if "BLOCK" in decision.action else "ALERT" if "ALERT" in decision.action else "ALLOW"
        self._records[txn_id] = {"txn": txn, "action": action_type, "risk_score": decision.risk_score,
                                 "agent_scores": dict(decision.agent_scores), "routing": routing}
        if len(self._records) > 500:
            self._records.pop(next(iter(self._records)))
        trace = self._build_trace(raw_transaction, txn, routing, results, decision, action_type)
        return {"transaction": txn, "routing": routing, "results": results,
                "decision": decision, "latency_ms": total_ms, "trace": trace}

    # ── agentic loop: Reactive → Detection → Decision → Feedback(self-review) → Reactive … ──
    def _run_pass(self, raw, txn, overrides=None, force_all=False):
        txn_id  = txn["transaction_id"]
        routing = self.router.route(txn, overrides, force_all)
        names   = [routing.primary_agent] + routing.secondary_agents
        results = [self.agents[n].predict(txn) for n in names if n in self.agents]
        decision = self.risk_engine.aggregate(txn_id, results)
        action_type = "BLOCK" if "BLOCK" in decision.action else "ALERT" if "ALERT" in decision.action else "ALLOW"
        trace = self._build_trace(raw, txn, routing, results, decision, action_type)
        return routing, names, results, decision, action_type, trace

    def process_agentic(self, raw_transaction: Dict) -> Dict:
        """Autonomous multi-pass processing. Whenever the Feedback Agent doubts a decision it rewrites the
        Reactive Agent's rules (for this case) and the whole pipeline starts over — up to MAX_RERUNS times."""
        from core.feedback import MAX_RERUNS
        t0  = time.time()
        txn = self.normalizer.normalize(raw_transaction)
        txn_id, n_total = txn["transaction_id"], len(self.agents)
        passes, overrides, force_all = [], None, False
        for level in range(MAX_RERUNS + 1):
            routing, names, results, decision, action_type, trace = self._run_pass(raw_transaction, txn, overrides, force_all)
            rev = self.feedback.self_review(decision, action_type, len(names), n_total)
            entry = {"pass": level + 1, "action": action_type, "risk_score": decision.risk_score,
                     "agents_run": names, "agent_scores": dict(decision.agent_scores),
                     "trace": trace, "routing_reason": routing.reason,
                     "doubtful": rev["doubtful"], "reasons": rev["reasons"], "tests": rev["tests"],
                     "feedback_messages": list(rev["messages"]), "rewrites": [], "next": "final"}
            passes.append(entry)
            fm = entry["feedback_messages"]
            if not rev["doubtful"]:
                fm.append(f"VERDICT: confident after pass {level+1} → decision {action_type} is final. Sending it to Alert & Action.")
                break
            if len(names) >= n_total:
                fm.append(f"VERDICT: still doubtful, but all {n_total} agents have already inspected it — no new evidence can be gathered. "
                          f"Final decision stands: {action_type}.")
                break
            if level == MAX_RERUNS:
                fm.append(f"VERDICT: still doubtful after {level+1} passes (re-run limit reached). Final decision stands: {action_type}.")
                break
            rewrites, overrides, force_all, adding = self.feedback.escalate(names, txn, overrides)
            entry["rewrites"], entry["next"], entry["adding"] = rewrites, "rerun", adding
            fm.append(f"VERDICT: DOUBTFUL → I need more evidence. Rewriting the Reactive Agent's rules so the {adding.replace('_AGENT','').title()} check joins, then starting over (pass {level+2}).")
            for c in rewrites:
                fm.append(f"RULE CHANGED → {c['label']}: {c['old']} → {c['new']}  ({c['why']})")
            fm.append("Rewritten rules sent back to the Reactive Agent ↩")
        total_ms = round((time.time() - t0) * 1000, 1)
        # book-keeping once, for the FINAL pass only
        self._audit_log.append({"transaction_id": txn_id, "amount": txn["amount"], "channel": txn["channel"],
                                "risk_score": decision.risk_score, "action": decision.action,
                                "agents_run": names, "agent_scores": decision.agent_scores,
                                "signals": decision.risk_signals, "total_ms": total_ms})
        self._records[txn_id] = {"txn": txn, "action": action_type, "risk_score": decision.risk_score,
                                 "agent_scores": dict(decision.agent_scores), "routing": routing}
        if len(self._records) > 500:
            self._records.pop(next(iter(self._records)))
        return {"transaction": txn, "routing": routing, "results": results, "decision": decision,
                "latency_ms": total_ms, "trace": trace, "passes": passes}

    def give_feedback(self, transaction_id: str, true_is_fraud: bool) -> Dict:
        rec = self._records.get(transaction_id)
        if rec is None:
            raise KeyError(f"Unknown transaction_id '{transaction_id}' (process it first)")
        return self.feedback.give_feedback(rec, true_is_fraud)

    # ── detailed per-agent narrative (shown on the dashboard) ─────
    _MODEL_NAMES = {"ATM_AGENT": "Random Forest", "BANK_AGENT": "Gradient Boosting",
                    "CREDIT_AGENT": "Random Forest", "GRAPH_AGENT": "Graph features + classifier"}

    def _build_trace(self, raw, txn, routing, results, decision, action) -> List[Dict]:
        inr = lambda v: f"₹{v:,.2f}"
        # 1 — Transaction-Level Agent
        m1 = [f"Got a {txn['channel']} payment of {inr(txn['amount'])} from account {txn['account_id']}.",
              f"Time: hour {txn['hour']}, {'weekend' if txn['is_weekend'] else 'weekday'}{', holiday' if txn['is_holiday'] else ''}.",
              f"Behaviour: {txn['login_attempts']} login try(s), took {txn['duration_seconds']:.0f}s, balance {inr(txn['balance'])}.",
              "Data cleaned and ready — passing to the Reactive Agent."]
        # 2 — Reactive Agent
        m2 = []
        for c in routing.rule_checks:
            m2.append(f"{label(c['rule'])}: {c['test']} → {'YES, ' + c['effect'].replace('_AGENT',' check').replace('+','add ') if c['fired'] else 'no'}")
        agents_run = [routing.primary_agent] + routing.secondary_agents
        names = ", ".join(a.replace("_AGENT", "") for a in agents_run)
        m2.append(f"Reactive decision: run these checks → {names}")
        # 3 — Detection Agent
        m3 = []
        for r in results:
            sig = [x for x in r.reasoning.get("risk_signals", []) if x]
            lvl = "HIGH" if r.fraud_probability >= 0.6 else "MEDIUM" if r.fraud_probability >= 0.3 else "LOW"
            m3.append(f"{r.agent_name.replace('_AGENT','')} check → risk {r.fraud_probability:.2f} ({lvl})"
                      f"{' · why: ' + ', '.join(s.replace('_',' ').lower() for s in sig) if sig else ' · nothing suspicious'}")
        top = max(results, key=lambda r: r.fraud_probability) if results else None
        if top:
            m3.append(f"Highest risk came from the {top.agent_name.replace('_AGENT','')} check ({top.fraud_probability:.2f}).")
        # 4 — Decision Agent
        m4 = [friendly(x) for x in decision.steps]
        # 5 — Alert & Action
        acts = {"BLOCK": ["Declining the transaction — funds are NOT released.",
                          "Freezing the account pending investigation.",
                          "Notifying the bank's fraud team and the customer (SMS/e-mail)."],
                "ALERT": ["Transaction held for verification.",
                          "Sending OTP / verification call to the customer.",
                          "Adding the transaction to the analyst review queue."],
                "ALLOW": ["Transaction approved and released.",
                          "No customer notification needed.",
                          "Account stays under normal monitoring."]}[action]
        m5 = [f"Final decision: {action}. Next:"] + acts
        stages = [
            {"id": 1, "agent": "Transaction-Level Agent", "icon": "📥", "messages": m1,
             "summary": f"Normalised {txn['transaction_id']} · {inr(txn['amount'])} · {txn['channel']}"},
            {"id": 2, "agent": "Reactive Agent", "icon": "⚡", "messages": m2,
             "summary": f"Running: {names}"},
            {"id": 3, "agent": "Detection Agent", "icon": "🧠", "messages": m3,
             "summary": " · ".join(f"{r.agent_name.replace('_AGENT','')} {r.fraud_probability*100:.0f}%" for r in results)},
            {"id": 4, "agent": "Decision / Response Agent", "icon": "⚖️", "messages": m4,
             "summary": f"Risk {decision.risk_score:.4f} → {action}"},
            {"id": 5, "agent": "Alert & Action", "icon": "🚨", "messages": m5, "summary": acts[0]},
        ]
        return stages

    def process_batch(self, transactions: List[Dict], verbose: bool = False) -> pd.DataFrame:
        rows = []
        for i, txn in enumerate(transactions):
            out = self.process(txn, verbose=verbose)
            d = out["decision"]
            rows.append({
                "idx": i, "transaction_id": d.transaction_id,
                "amount": out["transaction"]["amount"],
                "channel": out["transaction"]["channel"],
                "risk_score": d.risk_score, "action": d.action,
                "primary_agent": d.primary_agent,
                "signals": "|".join(d.risk_signals),
                "latency_ms": out["latency_ms"],
            })
        return pd.DataFrame(rows)

    def get_audit_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._audit_log)
