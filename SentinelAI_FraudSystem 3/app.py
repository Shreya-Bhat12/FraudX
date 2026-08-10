"""
app.py — Flask Backend API
SentinelAI Fraud Detection System
"""

import sys, os
# Always resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import json, time, random, threading
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'), static_url_path='')

# ── Dataset paths (edit these if your CSVs are elsewhere) ─────
ATM_DATA_PATH  = os.path.join(BASE_DIR, 'data', 'ATM_transaction_dataset.csv')
BANK_DATA_PATH = os.path.join(BASE_DIR, 'data', 'bank_transactions_data_2.csv')

# ── Global orchestrator ────────────────────────────────────────
orchestrator = None
_lock = threading.Lock()

def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        with _lock:
            if orchestrator is None:
                from fraud_orchestrator import FraudOrchestrator
                orch = FraudOrchestrator()
                orch.train_all(atm_path=ATM_DATA_PATH, bank_path=BANK_DATA_PATH)
                orchestrator = orch
    return orchestrator

def preload():
    print("⏳ Pre-loading models in background...")
    try:
        get_orchestrator()
        print("✅ Models ready.")
    except Exception as e:
        print(f"❌ Model load error: {e}")
        print(f"   Make sure CSVs are in: {os.path.join(BASE_DIR, 'data', '')}")

threading.Thread(target=preload, daemon=True).start()

# ═══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'index.html')

@app.route('/api/status')
def status():
    ready = orchestrator is not None
    return jsonify({
        "ready": ready,
        "message": "All agents online" if ready else "Loading models...",
        "agents": ["ATM_AGENT","BANK_AGENT","CREDIT_AGENT","GRAPH_AGENT"] if ready else [],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    orch = get_orchestrator()
    if not orch:
        return jsonify({"error": "Models not ready yet"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    try:
        result = orch.process(data, verbose=False)
        d   = result["decision"]
        txn = result["transaction"]
        action_type = "BLOCK" if "BLOCK" in d.action else "ALERT" if "ALERT" in d.action else "ALLOW"
        return jsonify({
            "transaction_id":   d.transaction_id,
            "risk_score":       d.risk_score,
            "action":           action_type,
            "action_label":     d.action,
            "primary_agent":    d.primary_agent,
            "agent_scores":     d.agent_scores,
            "agent_flags":      d.agent_flags,
            "risk_signals":     d.risk_signals,
            "recommended":      d.recommended_action,
            "confidence":       d.confidence,
            "latency_ms":       result["latency_ms"],
            "routing": {
                "primary":   result["routing"].primary_agent,
                "secondary": result["routing"].secondary_agents,
                "reason":    result["routing"].reason,
            },
            "normalized": {
                "amount":    txn.get("amount"),
                "channel":   txn.get("channel"),
                "location":  txn.get("location"),
                "hour":      txn.get("hour"),
                "is_weekend":txn.get("is_weekend"),
                "is_holiday":txn.get("is_holiday"),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch():
    orch = get_orchestrator()
    if not orch:
        return jsonify({"error": "Models not ready"}), 503
    data = request.get_json()
    transactions = data.get("transactions", [])
    if not transactions:
        return jsonify({"error": "No transactions provided"}), 400
    results = []
    for txn in transactions[:50]:
        try:
            r = orch.process(txn, verbose=False)
            d = r["decision"]
            action_type = "BLOCK" if "BLOCK" in d.action else "ALERT" if "ALERT" in d.action else "ALLOW"
            results.append({
                "transaction_id": d.transaction_id,
                "amount":         r["transaction"].get("amount"),
                "channel":        r["transaction"].get("channel"),
                "risk_score":     d.risk_score,
                "action":         action_type,
                "agent_scores":   d.agent_scores,
                "signals":        d.risk_signals,
                "latency_ms":     r["latency_ms"],
            })
        except Exception as e:
            results.append({"error": str(e), "transaction_id": txn.get("transaction_id","?")})
    summary = {
        "total":   len(results),
        "blocked": sum(1 for r in results if r.get("action")=="BLOCK"),
        "alerted": sum(1 for r in results if r.get("action")=="ALERT"),
        "allowed": sum(1 for r in results if r.get("action")=="ALLOW"),
        "avg_risk": round(sum(r.get("risk_score",0) for r in results)/max(len(results),1), 4),
        "avg_latency_ms": round(sum(r.get("latency_ms",0) for r in results)/max(len(results),1), 1),
    }
    return jsonify({"results": results, "summary": summary})

@app.route('/api/simulate')
def simulate():
    channels  = ["ATM","online","upi","transfer","cheque","branch"]
    locations = ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Airport ATM","Big Street ATM","Unknown"]
    accts     = [f"AC{str(i).zfill(5)}" for i in range(1,20)]
    merchants = [f"M{str(i).zfill(3)}" for i in range(1,30)]
    fraud_mode = request.args.get("fraud","false").lower() == "true"
    if fraud_mode:
        txn = {
            "transaction_id":   f"SIM{int(time.time()*1000)%100000}",
            "account_id":       random.choice(accts),
            "amount":           round(random.uniform(900, 1900), 2),
            "TransactionDate":  datetime.now().replace(hour=random.randint(0,5)).isoformat(),
            "channel":          random.choice(["online","upi","transfer"]),
            "location":         random.choice(["Unknown","Suspicious"]),
            "login_attempts":   random.randint(3, 5),
            "balance":          round(random.uniform(100, 500), 2),
            "duration_seconds": random.randint(5, 20),
            "customer_age":     random.randint(18, 30),
            "merchant_id":      "M999",
        }
    else:
        txn = {
            "transaction_id":   f"SIM{int(time.time()*1000)%100000}",
            "account_id":       random.choice(accts),
            "amount":           round(random.uniform(50, 600), 2),
            "TransactionDate":  datetime.now().replace(hour=random.randint(9,18)).isoformat(),
            "channel":          random.choice(channels),
            "location":         random.choice(locations[:-2]),
            "login_attempts":   1,
            "balance":          round(random.uniform(2000, 15000), 2),
            "duration_seconds": random.randint(40, 200),
            "customer_age":     random.randint(25, 65),
            "merchant_id":      random.choice(merchants),
        }
    return jsonify(txn)

@app.route('/api/stats')
def stats():
    orch = get_orchestrator()
    if not orch:
        return jsonify({"ready": False, "total_processed": 0})
    audit = orch.get_audit_log()
    if len(audit) == 0:
        return jsonify({"total_processed":0,"fraud_rate":0,"blocked":0,"alerted":0,"allowed":0,
                        "avg_risk":0,"avg_latency":0,"channel_breakdown":{},"recent":[]})
    blocked = int(audit["action"].str.contains("BLOCK").sum())
    alerted = int(audit["action"].str.contains("ALERT").sum())
    allowed = int(audit["action"].str.contains("ALLOW").sum())
    return jsonify({
        "total_processed": len(audit),
        "fraud_rate":       round((blocked+alerted)/max(len(audit),1)*100, 1),
        "blocked":          blocked,
        "alerted":          alerted,
        "allowed":          allowed,
        "avg_risk":         round(float(audit["risk_score"].mean()), 4),
        "avg_latency":      round(float(audit["total_ms"].mean()), 1),
        "channel_breakdown": audit["channel"].value_counts().to_dict(),
        "recent": audit.tail(10)[["transaction_id","amount","channel","risk_score","action"]].to_dict("records")
    })

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🏦 FRAUD DETECTION SYSTEM — Starting Server")
    print("="*55)
    print("  URL: http://localhost:5050")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
