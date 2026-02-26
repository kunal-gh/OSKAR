import os
import json
import uuid
import pytest
from pathlib import Path
from audit_logger import AuditLogger, AUDIT_LOG_FILE

def test_audit_logger_creates_file():
    """Ensure the logger creates the directory and file if it doesn't exist."""
    req_id = str(uuid.uuid4())
    
    entry = AuditLogger.log_decision(
        request_id=req_id,
        user_id="test_user",
        actor_role="analyst",
        text_hash="dummy_hash",
        route_decision="human_review",
        bot_score=0.88,
        hate_score=0.99,
        trust_score=0.2,
        final_risk_score=0.95
    )
    
    assert AUDIT_LOG_FILE.exists()
    
    # Read the last line to verify format
    with open(AUDIT_LOG_FILE, "r") as f:
        lines = f.readlines()
        last_line = json.loads(lines[-1].strip())
        
    assert last_line["request_id"] == req_id
    assert last_line["actor_role"] == "analyst"
    assert last_line["bot_score"] == 0.88
    assert last_line["final_risk_score"] == 0.95
    assert "timestamp" in last_line
