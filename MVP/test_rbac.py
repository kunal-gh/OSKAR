import pytest
from fastapi.testclient import TestClient
from main import app
from auth_manager import API_KEYS, UserRole, LEGACY_KEY

client = TestClient(app)

def test_rbac_analyst_can_analyze():
    """An Analyst can access the /analyze endpoints."""
    # Note: the actual analyze endpoint requires ML models to be loaded.
    # To avoid triggering a heavy test, we'll just check if it gets past auth.
    # A 422 Unprocessable Entity means it passed auth but failed validation. 
    # A 403 / 401 means it failed auth.
    
    response = client.post("/analyze", json={"user_id": "test", "text": "test"}, headers={"X-API-Key": LEGACY_KEY})
    # Auth passed, but it might hang or return 500 if models pending. We just ensure it's not 401/403.
    assert response.status_code not in [401, 403]

def test_rbac_analyst_cannot_access_reports():
    """An Analyst should be forbidden from accessing admin-only reporting endpoints."""
    response = client.get("/warning/report", headers={"X-API-Key": LEGACY_KEY})
    assert response.status_code == 403
    assert "Requires ADMIN role" in response.json().get("detail", "")

def test_rbac_admin_can_access_reports():
    """An Admin can access the reporting endpoints."""
    admin_key = "oskar-admin-key-999"
    response = client.get("/warning/report", headers={"X-API-Key": admin_key})
    # Should be 200 OK since admin has access
    assert response.status_code == 200

def test_rbac_missing_api_key():
    """Missing key should return 401 Unauthorized."""
    response = client.get("/warning/report")
    assert response.status_code == 401

def test_rbac_invalid_api_key():
    """Invalid key should return 403 Forbidden."""
    response = client.get("/warning/report", headers={"X-API-Key": "fake-hacker-key"})
    assert response.status_code == 403
