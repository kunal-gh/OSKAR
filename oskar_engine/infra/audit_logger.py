import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Ensure logs directory exists
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
AUDIT_LOG_FILE = LOGS_DIR / "decision_audit.log"

# Optional: Also configure standard Python logging to stream to console
logger = logging.getLogger("AuditLogger")
logger.setLevel(logging.INFO)


class AuditLogger:
    """
    Immutable, append-only JSONL logger for Enterprise compliance auditing.
    Records the variables and math that led to a specific routing decision.
    """

    @staticmethod
    def log_decision(
        request_id: str,
        user_id: str,
        actor_role: str,
        text_hash: str,
        route_decision: str,
        bot_score: float,
        hate_score: float,
        trust_score: float,
        final_risk_score: float,
    ):
        """Append a structured decision record to the audit log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "user_id": user_id,
            "actor_role": actor_role,
            "text_hash": text_hash,
            "route_decision": route_decision,
            "bot_score": round(bot_score, 4),
            "hate_score": round(hate_score, 4),
            "trust_score": round(trust_score, 4),
            "final_risk_score": round(final_risk_score, 4),
        }

        # Append-only JSON line
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry
