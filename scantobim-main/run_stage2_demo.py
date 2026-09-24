#!/usr/bin/env python3
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "ScanTOBIM" / "ScanTOBIM" / "run_stage2_demo.py"
if not target.exists():
    sys.exit(f"Target not found: {target}")

sys.path.insert(0, str(target.parent))
with open(target) as f:
    code = compile(f.read(), str(target), 'exec')
    exec(code)
