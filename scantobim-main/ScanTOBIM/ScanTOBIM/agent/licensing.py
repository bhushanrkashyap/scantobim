"""ScanToBIM Licensing & Feature Flags.

Tiers:
  starter    — up to 3 sessions/month, IFC export, no Revit bridge
  pro        — unlimited sessions, IFC + Revit bridge, clash detection
  enterprise — everything + multi-tenant, SSO, on-prem deploy, audit export

Tier is encoded in the JWT claim `tier` (set at login / provisioning).
Feature checks are done via `require_feature()` FastAPI dependency.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from pydantic import BaseModel

from agent.auth import CurrentUser, get_current_user

# ── Feature catalogue ─────────────────────────────────────────────────────────

FEATURES: dict[str, dict[str, bool | int]] = {
    # feature_key → {starter, pro, enterprise}
    "ifc_export": {"starter": True, "pro": True, "enterprise": True},
    "revit_bridge": {"starter": False, "pro": True, "enterprise": True},
    "clash_detection": {"starter": False, "pro": True, "enterprise": True},
    "coordinate_alignment": {"starter": True, "pro": True, "enterprise": True},
    "ncr_management": {"starter": True, "pro": True, "enterprise": True},
    "cde_publish": {"starter": False, "pro": True, "enterprise": True},
    "audit_export": {"starter": False, "pro": False, "enterprise": True},
    "multi_tenant": {"starter": False, "pro": False, "enterprise": True},
    "sso": {"starter": False, "pro": False, "enterprise": True},
    "sse_streaming": {"starter": True, "pro": True, "enterprise": True},
    "scan_upload": {"starter": True, "pro": True, "enterprise": True},
    "api_access": {"starter": True, "pro": True, "enterprise": True},
    # Numeric limits
    "max_sessions_per_month": {"starter": 3, "pro": 9999, "enterprise": 9999},
    "max_scan_size_mb": {"starter": 200, "pro": 2000, "enterprise": 10000},
    "max_elements_per_session": {"starter": 500, "pro": 5000, "enterprise": 50000},
}

# ── Tier helpers ──────────────────────────────────────────────────────────────

# Tenant → tier mapping (replace with DB lookup in Sprint 4)
# admin and demo tenants default to enterprise for development
_TENANT_TIERS: dict[str, str] = {
    "system": "enterprise",
    "demo": "enterprise",  # demo gets full access for PoV
}
_DEFAULT_TIER = "starter"


def get_tenant_tier(tenant_id: str) -> str:
    """Return the tier for a tenant ID."""
    return _TENANT_TIERS.get(tenant_id, _DEFAULT_TIER)


def feature_enabled(tenant_id: str, feature: str) -> bool:
    """Check if a boolean feature is enabled for a tenant's tier."""
    tier = get_tenant_tier(tenant_id)
    feat = FEATURES.get(feature, {})
    value = feat.get(tier, False)
    return bool(value)


def feature_limit(tenant_id: str, feature: str) -> int:
    """Return the numeric limit for a feature for a tenant's tier."""
    tier = get_tenant_tier(tenant_id)
    feat = FEATURES.get(feature, {})
    value = feat.get(tier, 0)
    return int(value)


# ── FastAPI dependency ────────────────────────────────────────────────────────


def require_feature(feature_key: str):
    """Dependency factory: block endpoint if feature not in tenant's tier.

    Usage:
        @app.get("/export/ifc")
        async def export_ifc(user = Depends(require_feature("ifc_export"))):
            ...
    """

    def _checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not feature_enabled(user.tenant_id, feature_key):
            tier = get_tenant_tier(user.tenant_id)
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Feature '{feature_key}' is not available on the '{tier}' tier. "
                    f"Upgrade to Pro or Enterprise to unlock this feature."
                ),
            )
        return user

    return _checker


# ── License info model ────────────────────────────────────────────────────────


class LicenseInfo(BaseModel):
    tenant_id: str
    tier: str
    features: dict[str, bool | int]

    @classmethod
    def for_tenant(cls, tenant_id: str) -> LicenseInfo:
        tier = get_tenant_tier(tenant_id)
        resolved: dict[str, bool | int] = {}
        for key, tiers in FEATURES.items():
            val = tiers.get(tier, False)
            resolved[key] = int(val) if isinstance(val, bool) is False else bool(val)
        return cls(tenant_id=tenant_id, tier=tier, features=resolved)
