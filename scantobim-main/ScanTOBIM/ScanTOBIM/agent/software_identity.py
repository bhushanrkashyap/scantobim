"""software_identity.py — NQA-1 Subpart 2.7 § 302 "Software Identification".

Every audit event + every deliverable must pin the exact software version
that produced it, so a reviewer can reproduce the result from archived code.

Records captured:
  • Git commit SHA (HEAD)
  • Git tree clean / dirty flag
  • Git tag (if HEAD is tagged)
  • Build timestamp (UTC)
  • Python version
  • Platform (OS + arch)
  • Locked dependency versions for packages that materially affect output
    (open3d, numpy, scipy, laspy, pydantic)

This module is intentionally dependency-free beyond stdlib + importlib.metadata
so it loads early in startup and cannot itself become a compliance risk.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path

# Packages whose version materially affects numerical output.
# Listed in the order they appear in the pipeline.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "open3d",
    "open3d-cpu",
    "laspy",
    "pydantic",
    "fastapi",
    "structlog",
)


@dataclass(frozen=True)
class SoftwareIdentity:
    product_name: str
    product_version: str
    git_sha: str
    git_tag: str | None
    git_tree_clean: bool
    built_at_utc: str
    python_version: str
    platform_label: str
    dependencies: dict[str, str]
    identity_hash: str  # SHA-256 of the above (excl. built_at_utc) — stable ID

    def to_dict(self) -> dict:
        return asdict(self)


def _run_git(args: list[str], repo_root: Path) -> str | None:
    """Run git with args; return stripped stdout or None on failure."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _project_root() -> Path:
    """Walk up from this file until we find a dir with .git/ or pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return here.parent


def _read_version() -> str:
    """Read the product version from CHANGELOG.md (first '## [x.y.z]' heading).

    Falls back to 'dev' if the file is not present or unparseable.
    """
    root = _project_root()
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        return "dev"
    try:
        text = changelog.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("## [") and "]" in line:
                inside = line.split("[", 1)[1].split("]", 1)[0]
                return inside if inside else "dev"
    except Exception:
        pass
    return "dev"


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _compute_identity_hash(payload: dict) -> str:
    """Deterministic SHA-256 of the identity fields (minus the timestamp)."""
    stable = {k: v for k, v in payload.items() if k not in ("built_at_utc", "identity_hash")}
    serialised = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def get_software_identity() -> SoftwareIdentity:
    """Resolve the current software identity. Cached — computed once per process."""
    root = _project_root()

    sha = _run_git(["rev-parse", "HEAD"], root) or "unknown"
    tag = _run_git(["describe", "--exact-match", "--tags", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    tree_clean = status is not None and status == ""

    payload = {
        "product_name": "ScanToBIM",
        "product_version": _read_version(),
        "git_sha": sha,
        "git_tag": tag,
        "git_tree_clean": tree_clean,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform_label": f"{platform.system()} {platform.machine()}",
        "dependencies": _dependency_versions(),
    }
    payload["identity_hash"] = _compute_identity_hash(payload)

    return SoftwareIdentity(**payload)


def software_identity_dict() -> dict:
    """Convenience — callers that want a plain dict without importing the class."""
    return get_software_identity().to_dict()


def clear_cache() -> None:
    """Reset the identity cache. Useful in tests that patch git state."""
    get_software_identity.cache_clear()
