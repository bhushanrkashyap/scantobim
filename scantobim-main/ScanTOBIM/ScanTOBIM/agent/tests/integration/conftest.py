"""
conftest.py for agent/tests/integration/

Integration tests in this subdirectory use the REAL open3d library,
not the project-level mock injected by agent/tests/conftest.py.

The project conftest runs first and injects the mock; this conftest
immediately removes those mock entries so real open3d is importable.
"""

import sys

# Remove mock open3d injected by the parent conftest so real open3d loads.
# This must happen before any test module in this directory imports open3d.
for _mod in list(sys.modules):
    if _mod == "open3d" or _mod.startswith("open3d."):
        del sys.modules[_mod]
