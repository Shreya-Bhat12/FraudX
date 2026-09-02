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
        return {"transaction": txn, "routing": routing, "results": results,
                "decision": decision, "latency_ms": total_ms}

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
