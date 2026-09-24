"""CUDA Dependency Guard — Phase 3A CPU Production.

Engineering Rule:
  Production code MUST NOT import torch.cuda, call .cuda(), .to('cuda'),
  or depend on CUDA-only custom operators.

This module provides:
  - scan_for_cuda_imports(): Statically scan source files for CUDA violations.
  - assert_no_cuda(): Runtime assertion that CUDA is not being used.
  - CUDAProductionViolation: Exception raised on violation.

Usage in CI:
  python3 -m agent.phase3.neural.cuda_guard --scan agent/
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Patterns that indicate CUDA dependency in production code
_CUDA_VIOLATION_PATTERNS: list[str] = [
    r"\.cuda\(\)",
    r'\.to\(["\']cuda',
    r"torch\.cuda\b",
    r"import torch\.cuda",
    r"from torch\.cuda",
    r"torch\.backends\.cudnn",
    r"CUDA_VISIBLE_DEVICES",
    r"nvcc\b",
    r"\.half\(\)",          # FP16 is GPU-specific in practice
    r"torch\.float16",
    r"spconv",              # Sparse GPU convolution
    r"MinkowskiEngine",     # GPU sparse convolution
]

_CUDA_VIOLATION_COMPILED = [re.compile(p) for p in _CUDA_VIOLATION_PATTERNS]

# Files/directories that are allowed to contain GPU code (research isolation)
_ALLOWED_GPU_PATHS: list[str] = [
    "research_gpu_optional",
    "tests/",               # Tests may probe CUDA availability
    "cuda_guard.py",        # This file itself
    "ptv2_adapter.py",      # Neural adapter stub — may document CUDA req
    "kpconv_adapter.py",    # Neural adapter stub — may document CUDA req
]


class CUDAProductionViolation(RuntimeError):
    """Raised when production code contains a CUDA dependency."""


def is_allowed_gpu_path(path: Path) -> bool:
    """Return True if this path is in the research/test isolation zone."""
    path_str = str(path)
    return any(allowed in path_str for allowed in _ALLOWED_GPU_PATHS)


def scan_file_for_cuda(path: Path) -> list[dict[str, object]]:
    """Scan a single Python file for CUDA violation patterns.

    Returns a list of violation dicts: {file, line_number, line, pattern}.
    """
    if is_allowed_gpu_path(path):
        return []

    violations: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip pure comments
        if stripped.startswith("#"):
            continue
        for pattern_re in _CUDA_VIOLATION_COMPILED:
            if pattern_re.search(line):
                violations.append({
                    "file": str(path),
                    "line_number": lineno,
                    "line": line.rstrip(),
                    "pattern": pattern_re.pattern,
                })
                break  # one violation per line is enough
    return violations


def scan_for_cuda_imports(root: Path | str) -> list[dict[str, object]]:
    """Recursively scan all Python files under root for CUDA violations.

    Args:
        root: Directory to scan.

    Returns:
        List of violation dicts.
    """
    root = Path(root)
    all_violations: list[dict[str, object]] = []

    for py_file in sorted(root.rglob("*.py")):
        # Skip __pycache__
        if "__pycache__" in str(py_file):
            continue
        violations = scan_file_for_cuda(py_file)
        all_violations.extend(violations)

    return all_violations


def assert_no_cuda_runtime() -> None:
    """Runtime assertion: CUDA must not be active or imported.

    Call at application startup for CPU-only deployments.

    Raises:
        CUDAProductionViolation if CUDA is somehow active.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # CUDA being available is not a violation — using it is.
            # Just log a warning.
            logger.warning(
                "cuda_available_on_system",
                note="CUDA is available but MUST NOT be used in production path.",
            )
        # Check no CUDA tensors exist
        if torch.cuda.is_initialized():
            raise CUDAProductionViolation(
                "CUDA context is initialized — production code must use CPU only."
            )
    except ImportError:
        pass  # PyTorch not installed: no CUDA possible


def assert_tensor_on_cpu(tensor_or_module: object, name: str = "tensor") -> None:
    """Assert that a PyTorch tensor or module is on CPU.

    Raises:
        CUDAProductionViolation if the object is on a CUDA device.
    """
    try:
        import torch
        if isinstance(tensor_or_module, torch.Tensor):
            if tensor_or_module.device.type != "cpu":
                raise CUDAProductionViolation(
                    f"Production tensor '{name}' is on device "
                    f"'{tensor_or_module.device}' — must be 'cpu'."
                )
        elif isinstance(tensor_or_module, torch.nn.Module):
            for param_name, param in tensor_or_module.named_parameters():
                if param.device.type != "cpu":
                    raise CUDAProductionViolation(
                        f"Production model parameter '{param_name}' is on device "
                        f"'{param.device}' — must be 'cpu'."
                    )
    except ImportError:
        pass


def get_production_device() -> str:
    """Return the authoritative production device string.

    Always 'cpu'. Never 'cuda'. Never 'mps' for production.
    """
    return "cpu"


if __name__ == "__main__":
    # CLI: python3 -m agent.phase3.neural.cuda_guard --scan <path>
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Scan for CUDA violations in production code.")
    parser.add_argument("--scan", required=True, help="Root directory to scan.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    args = parser.parse_args()

    violations = scan_for_cuda_imports(args.scan)

    if args.json:
        print(json.dumps(violations, indent=2))
    else:
        if violations:
            print(f"\n❌ CUDA VIOLATIONS FOUND ({len(violations)}):\n")
            for v in violations:
                print(f"  {v['file']}:{v['line_number']}: {v['line']}")
            sys.exit(1)
        else:
            print("✅ No CUDA violations in production code.")
            sys.exit(0)
