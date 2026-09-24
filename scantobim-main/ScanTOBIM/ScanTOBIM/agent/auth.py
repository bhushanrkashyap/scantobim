"""JWT authentication + RBAC for ScanToBIM Agent.

Roles (least → most privileged):
  surveyor      — upload scans, view own sessions
  engineer      — approve SC2 safety gates
  bim_manager   — download IFC, manage CDE states
  qa_inspector  — raise / resolve NCRs
  admin         — full access, manage users, view audit chain

Token flow:
  POST /auth/token  → {access_token, token_type}
  Authorization: Bearer <token>  on all protected routes
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

log = structlog.get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

JWT_SECRET: str = os.environ.get("JWT_SECRET", "change-me-in-production-32-chars!!!")
JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))

# ── Role definitions ──────────────────────────────────────────────────────────

ROLES: list[str] = [
    "surveyor",
    "engineer",
    "bim_manager",
    "qa_inspector",
    "admin",
]

# Role → allowed actions (resource:action pairs)
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "surveyor": {
        "session:create",
        "session:read",
        "scan:upload",
        "progress:read",
    },
    "engineer": {
        "session:read",
        "gate:read",
        "gate:approve",
        "gate:reject",
    },
    "bim_manager": {
        "session:read",
        "session:list",
        "ifc:export",
        "cde:transition",
        "ncr:read",
    },
    "qa_inspector": {
        "session:read",
        "ncr:read",
        "ncr:raise",
        "ncr:resolve",
        "audit:read",
    },
    "admin": {
        # admin inherits everything — checked separately
        "*",
    },
}

# ── In-memory user store (replace with DB in Sprint 4) ───────────────────────
# Format: username → {hashed_password, role, tenant_id}
# Default dev users — CHANGE PASSWORDS before production!


def _hash_pw(pw: str) -> str:
    """SHA-256 password hash (upgrade to bcrypt in Sprint 4)."""
    return hashlib.sha256(pw.encode()).hexdigest()


_USERS: dict[str, dict] = {
    "admin": {
        "hashed_password": _hash_pw("admin123"),
        "role": "admin",
        "tenant_id": "system",
        "display_name": "System Administrator",
    },
    "surveyor": {
        "hashed_password": _hash_pw("survey123"),
        "role": "surveyor",
        "tenant_id": "demo",
        "display_name": "Survey Technician",
    },
    "engineer": {
        "hashed_password": _hash_pw("engineer123"),
        "role": "engineer",
        "tenant_id": "demo",
        "display_name": "Structural Engineer",
    },
    "bim": {
        "hashed_password": _hash_pw("bim123"),
        "role": "bim_manager",
        "tenant_id": "demo",
        "display_name": "BIM Coordinator",
    },
    "qa": {
        "hashed_password": _hash_pw("qa123"),
        "role": "qa_inspector",
        "tenant_id": "demo",
        "display_name": "QA Inspector",
    },
}

# ── Pydantic models ───────────────────────────────────────────────────────────


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_MINUTES * 60
    role: str
    tenant_id: str


class CurrentUser(BaseModel):
    username: str
    role: str
    tenant_id: str
    display_name: str


# ── Minimal JWT implementation (no external jwt lib needed) ───────────────────
# Uses HMAC-SHA256 — same pattern as the audit ledger.
# Structure: base64url(header).base64url(payload).base64url(sig)

import base64
import json


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def create_access_token(username: str, role: str, tenant_id: str) -> str:
    """Create a signed JWT access token."""
    now = int(time.time())
    header = _b64url(json.dumps({"alg": JWT_ALGORITHM, "typ": "JWT"}).encode())
    payload = _b64url(
        json.dumps(
            {
                "sub": username,
                "role": role,
                "tenant_id": tenant_id,
                "iat": now,
                "exp": now + JWT_EXPIRE_MINUTES * 60,
            }
        ).encode()
    )
    signing_input = f"{header}.{payload}"
    sig = _b64url(
        hmac.new(
            JWT_SECRET.encode(),
            signing_input.encode(),
            hashlib.sha256,
        ).digest()
    )
    return f"{signing_input}.{sig}"


def decode_token(token: str) -> dict:
    """Decode and verify JWT. Raises HTTPException on failure."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("malformed token")
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = _b64url(
            hmac.new(
                JWT_SECRET.encode(),
                signing_input.encode(),
                hashlib.sha256,
            ).digest()
        )
        if not hmac.compare_digest(sig_b64, expected_sig):
            raise ValueError("invalid signature")
        payload = json.loads(_b64url_decode(payload_b64))
        if payload.get("exp", 0) < time.time():
            raise ValueError("token expired")
        return payload
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ── FastAPI dependency ────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> CurrentUser:
    """FastAPI dependency: validate token, return CurrentUser."""
    payload = decode_token(token)
    username: str = payload.get("sub", "")
    user_record = _USERS.get(username)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="user not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return CurrentUser(
        username=username,
        role=user_record["role"],
        tenant_id=user_record["tenant_id"],
        display_name=user_record["display_name"],
    )


def require_role(*allowed_roles: str):
    """Dependency factory: restrict endpoint to specific roles.

    Usage:
        @app.post("/admin-only")
        async def endpoint(user: CurrentUser = Depends(require_role("admin"))):
            ...
    """

    def _checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role == "admin":
            return user  # admin bypasses all role checks
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role '{user.role}' is not authorised for this action. "
                f"Required: {list(allowed_roles)}",
            )
        return user

    return _checker


def has_permission(user: CurrentUser, action: str) -> bool:
    """Check if user has a specific permission action."""
    if user.role == "admin":
        return True
    perms = ROLE_PERMISSIONS.get(user.role, set())
    return action in perms


# ── Auth helper for login endpoint ────────────────────────────────────────────


def authenticate_user(username: str, password: str) -> dict | None:
    """Verify credentials. Returns user record or None."""
    user = _USERS.get(username)
    if not user:
        return None
    if not hmac.compare_digest(user["hashed_password"], _hash_pw(password)):
        return None
    return user
