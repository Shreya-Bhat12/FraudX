"""
core/base_agent.py
──────────────────────────────────────────────────────────────────
Base class every agent inherits from.
Gives each agent: logging, status tracking, inter-agent messaging,
and a standard predict() interface.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class AgentStatus(Enum):
    IDLE     = "idle"
    RUNNING  = "running"
    DONE     = "done"
    ERROR    = "error"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender:      str
    receiver:    str
    payload:     Dict[str, Any]
    msg_id:      str  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:   float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Standardised result every agent returns."""
    agent_name:       str
    fraud_probability: float          # 0.0 – 1.0
    fraud_flag:       bool
    confidence:       float           # model confidence
    features_used:    List[str]
    reasoning:        Dict[str, Any]  # explainability dict
    latency_ms:       float
    status:           AgentStatus = AgentStatus.DONE


class BaseAgent(ABC):
    """
    Abstract base for all fraud detection agents.

    Every concrete agent must implement:
        _extract_features(transaction) → np.ndarray
        _run_model(features)           → (fraud_prob, confidence)
        _explain(features, prob)       → dict
    """

    def __init__(self, name: str, fraud_type: str):
        self.name        = name
        self.fraud_type  = fraud_type
        self.status      = AgentStatus.IDLE
        self.call_count  = 0
        self._inbox: List[AgentMessage] = []
        self._log:   List[str]          = []

    # ── public interface ───────────────────────────────────────
    def predict(self, transaction: Dict) -> AgentResult:
        """Main entry-point. Wraps _run with timing + status."""
        self.status = AgentStatus.RUNNING
        t0 = time.time()
        try:
            features      = self._extract_features(transaction)
            prob, conf    = self._run_model(features)
            reasoning     = self._explain(transaction, features, prob)
            flag          = prob >= self._threshold()
            result = AgentResult(
                agent_name        = self.name,
                fraud_probability = round(float(prob), 4),
                fraud_flag        = bool(flag),
                confidence        = round(float(conf), 4),
                features_used     = self._feature_names(),
                reasoning         = reasoning,
                latency_ms        = round((time.time() - t0) * 1000, 2),
                status            = AgentStatus.DONE,
            )
            self.status     = AgentStatus.DONE
            self.call_count += 1
            self._log.append(
                f"[{self.name}] txn={transaction.get('transaction_id','?')} "
                f"prob={prob:.3f} flag={flag}"
            )
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            self._log.append(f"[{self.name}] ERROR: {e}")
            return AgentResult(
                agent_name=self.name, fraud_probability=0.5,
                fraud_flag=False, confidence=0.0,
                features_used=[], reasoning={"error": str(e)},
                latency_ms=(time.time()-t0)*1000,
                status=AgentStatus.ERROR,
            )

    def receive(self, msg: AgentMessage):
        """Accept a message from another agent."""
        self._inbox.append(msg)

    def send(self, receiver_name: str, payload: dict) -> AgentMessage:
        """Create a message to send to another agent."""
        return AgentMessage(sender=self.name, receiver=receiver_name, payload=payload)

    def get_logs(self) -> List[str]:
        return self._log.copy()

    # ── to be implemented ──────────────────────────────────────
    @abstractmethod
    def _extract_features(self, transaction: Dict) -> Any:
        pass

    @abstractmethod
    def _run_model(self, features: Any):
        """Returns (fraud_probability: float, confidence: float)"""
        pass

    @abstractmethod
    def _explain(self, transaction: Dict, features: Any, prob: float) -> Dict:
        pass

    @abstractmethod
    def _feature_names(self) -> List[str]:
        pass

    def _threshold(self) -> float:
        return 0.5   # override per agent if needed
