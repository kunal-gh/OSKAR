"""
test_api_security.py — OSKAR v0.6
Tests X-API-Key authentication and PII pipeline integration.
"""

import pytest
from fastapi.testclient import TestClient
import os
from main import app

# We set a dummy API key for testing
os.environ["OSKAR_API_KEY"] = "test-secret-key"

client = TestClient(app)

def test_missing_api_key_rejected():
    """Requests without X-API-Key must be rejected with 401."""
    response = client.post("/analyze", json={
        "user_id": "u1",
        "text": "Some text"
    })
    assert response.status_code == 401
    assert "Invalid or missing API Key" in response.text


def test_invalid_api_key_rejected():
    """Requests with wrong X-API-Key must be rejected with 401."""
    response = client.post("/analyze", headers={"X-API-Key": "wrong-key"}, json={
        "user_id": "u1",
        "text": "Some text"
    })
    assert response.status_code == 401


def test_valid_api_key_accepted():
    """Requests with correct X-API-Key proceed to actual endpoint handling."""
    # We expect this to either pass or fail due to model loading in TestClient,
    # but the auth layer should let it through (not 401/403).
    # Since health check is unprotected, let's test a protected route.
    response = client.post("/warning/impression", headers={"X-API-Key": "test-secret-key"}, json={
        "user_id": "u1",
        "risk_score": 0.5
    })
    
    # 200 means auth passed and endpoint executed successfully
    assert response.status_code == 200
    assert "event_id" in response.json()
