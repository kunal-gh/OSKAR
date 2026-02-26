import os
from enum import Enum
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    SYSTEM = "system"

# Legacy key from v0.6 for backward compatibility with Dashboard
LEGACY_KEY = "REDACTED_USE_ENV_VAR"

# In-memory RBAC Database (In production, load from PostgreSQL/Vault)
API_KEYS = {
    "oskar-admin-key-999": UserRole.ADMIN,
    LEGACY_KEY: UserRole.ANALYST,
    "oskar-system-key-777": UserRole.SYSTEM
}

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def _get_role(api_key: str) -> UserRole:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    role = API_KEYS.get(api_key)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    return role

def require_analyst(api_key_header: str = Security(api_key_header)) -> UserRole:
    """Allows ANALYST, ADMIN, and SYSTEM roles."""
    role = _get_role(api_key_header)
    # All roles are permitted to perform analysis
    return role

def require_admin(api_key_header: str = Security(api_key_header)) -> UserRole:
    """Strictly allows ADMIN roles only."""
    role = _get_role(api_key_header)
    if role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Requires ADMIN role. Your role is {role.value}",
        )
    return role

def require_system(api_key_header: str = Security(api_key_header)) -> UserRole:
    """Allows SYSTEM and ADMIN roles for automated ingestion tasks."""
    role = _get_role(api_key_header)
    if role not in [UserRole.SYSTEM, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Requires SYSTEM role. Your role is {role.value}",
        )
    return role
