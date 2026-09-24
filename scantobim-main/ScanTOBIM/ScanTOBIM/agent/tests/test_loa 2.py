"""test_loa.py — Tests for USIBD LOA v3.1 support (P2.2a)

Covers:
  • Tier classification (σ → LOA tier mapping at 2σ)
  • Plane, cylinder, AABB σ computation
  • LOAStatement model + to_dict
  • worse_tier aggregation
  • /sessions/{id}/loa-report endpoint
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import numpy as np
import pytest

from agent.models import LOAStatementModel
from agent.tools.loa_tools import (
    DEFAULT_SCANNER_ACCURACY_MM,
    LOATier,
    aabb_sigma_mm,
    build_loa_statement,
    classify_loa_tier,
    cylinder_sigma_mm,
    plane_sigma_mm,
    summarise_tier_distribution,
    tolerance_for_tier,
    worse_tier,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. Tier classification
# ═════════════════════════════════════════════════════════════════════════════


class TestClassifyLOATier:
    def test_perfect_fit_is_loa50(self):
        # σ=0 → 2σ=0 mm → fits LOA50 (±0.1 mm)
        assert classify_loa_tier(0.0) == LOATier.LOA50

    def test_tight_fit_is_loa50(self):
        # σ=0.04 → 2σ=0.08 mm ≤ 0.1 → LOA50
        assert classify_loa_tier(0.04) == LOATier.LOA50

    def test_precision_fit_is_loa40(self):
        # σ=0.4 → 2σ=0.8 mm ≤ 1.0 → LOA40
        assert classify_loa_tier(0.4) == LOATier.LOA40

    def test_construction_fit_is_loa30(self):
        # σ=2 → 2σ=4 mm ≤ 5 → LOA30
        assert classify_loa_tier(2.0) == LOATier.LOA30

    def test_design_fit_is_loa20(self):
        # σ=6 → 2σ=12 mm ≤ 15 → LOA20
        assert classify_loa_tier(6.0) == LOATier.LOA20

    def test_loose_fit_is_loa10(self):
        # σ=20 → 2σ=40 mm ≤ 50 → LOA10
        assert classify_loa_tier(20.0) == LOATier.LOA10

    def test_very_loose_still_loa10(self):
        # σ=100 → 2σ=200 > 50 — returns LOA10 (floor)
        assert classify_loa_tier(100.0) == LOATier.LOA10

    def test_boundary_5mm_is_loa30(self):
        # σ=2.5 → 2σ=5.0 → exactly LOA30 boundary
        assert classify_loa_tier(2.5) == LOATier.LOA30

    def test_boundary_just_over_5mm_is_loa20(self):
        # σ=2.51 → 2σ=5.02 → LOA20
        assert classify_loa_tier(2.51) == LOATier.LOA20

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            classify_loa_tier(-1.0)


class TestToleranceForTier:
    def test_loa10(self):
        assert tolerance_for_tier(LOATier.LOA10) == 50.0

    def test_loa20(self):
        assert tolerance_for_tier(LOATier.LOA20) == 15.0

    def test_loa30(self):
        assert tolerance_for_tier(LOATier.LOA30) == 5.0

    def test_loa40(self):
        assert tolerance_for_tier(LOATier.LOA40) == 1.0

    def test_loa50(self):
        assert tolerance_for_tier(LOATier.LOA50) == 0.1


class TestWorseTier:
    def test_same_tier_returns_same(self):
        assert worse_tier(LOATier.LOA30, LOATier.LOA30) == LOATier.LOA30

    def test_returns_less_accurate(self):
        assert worse_tier(LOATier.LOA50, LOATier.LOA20) == LOATier.LOA20

    def test_commutative(self):
        assert worse_tier(LOATier.LOA40, LOATier.LOA10) == worse_tier(LOATier.LOA10, LOATier.LOA40)
        assert worse_tier(LOATier.LOA40, LOATier.LOA10) == LOATier.LOA10


# ═════════════════════════════════════════════════════════════════════════════
# 2. σ computation
# ═════════════════════════════════════════════════════════════════════════════


class TestPlaneSigma:
    def test_perfect_plane_zero_sigma(self):
        """Points exactly on z=0 plane → σ = 0."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        plane = (0.0, 0.0, 1.0, 0.0)  # z = 0
        sigma = plane_sigma_mm(pts, plane)
        assert sigma == pytest.approx(0.0, abs=1e-6)

    def test_noisy_plane_nonzero_sigma(self):
        """Points with 10 mm Gaussian noise → σ ≈ 10 mm."""
        rng = np.random.default_rng(42)
        pts = np.column_stack(
            [
                rng.uniform(0, 1, 1000),
                rng.uniform(0, 1, 1000),
                rng.normal(0, 0.01, 1000),  # σ = 10 mm in z
            ]
        )
        sigma = plane_sigma_mm(pts, (0.0, 0.0, 1.0, 0.0))
        assert 8.0 <= sigma <= 12.0  # within ~20% of expected

    def test_zero_norm_returns_inf(self):
        pts = np.zeros((10, 3))
        sigma = plane_sigma_mm(pts, (0.0, 0.0, 0.0, 0.0))
        assert np.isinf(sigma)


class TestCylinderSigma:
    def test_perfect_cylinder_zero_sigma(self):
        """Points exactly on radius=0.1 around z-axis → σ = 0."""
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        heights = np.linspace(0, 1, 20)
        pts = np.column_stack(
            [
                0.1 * np.cos(angles),
                0.1 * np.sin(angles),
                heights,
            ]
        )
        sigma = cylinder_sigma_mm(
            pts,
            axis_point_m=np.array([0.0, 0.0, 0.5]),
            axis_direction=np.array([0.0, 0.0, 1.0]),
            radius_m=0.1,
        )
        assert sigma == pytest.approx(0.0, abs=1e-6)

    def test_noisy_cylinder(self):
        """Points with radial noise → σ ≈ noise scale."""
        rng = np.random.default_rng(7)
        n = 500
        angles = rng.uniform(0, 2 * np.pi, n)
        heights = rng.uniform(0, 1, n)
        radii = 0.1 + rng.normal(0, 0.005, n)  # σ = 5 mm
        pts = np.column_stack(
            [
                radii * np.cos(angles),
                radii * np.sin(angles),
                heights,
            ]
        )
        sigma = cylinder_sigma_mm(
            pts,
            axis_point_m=np.array([0.0, 0.0, 0.5]),
            axis_direction=np.array([0.0, 0.0, 1.0]),
            radius_m=0.1,
        )
        # σ of |N(0, 5mm)| ≈ 5·sqrt(1-2/π) ≈ 3 mm; loose bounds
        assert 1.0 <= sigma <= 8.0


class TestAABBSigma:
    def test_uniform_cube_gives_finite_sigma(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform(0, 1, (500, 3))
        sigma = aabb_sigma_mm(pts)
        assert sigma > 0
        assert np.isfinite(sigma)

    def test_empty_returns_inf(self):
        sigma = aabb_sigma_mm(np.empty((0, 3)))
        assert np.isinf(sigma)


# ═════════════════════════════════════════════════════════════════════════════
# 3. LOAStatement builder + aggregation
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildLOAStatement:
    def test_both_sigmas_combine_to_worst(self):
        """R=LOA30, M=LOA20 → effective LOA20."""
        s = build_loa_statement(
            segment_id="seg-1",
            represented_sigma_mm=2.0,  # LOA30
            measured_sigma_mm=6.0,  # LOA20
        )
        assert s.represented_tier == LOATier.LOA30
        assert s.measured_tier == LOATier.LOA20
        assert s.effective_tier == LOATier.LOA20

    def test_default_measured_accuracy(self):
        s = build_loa_statement("seg-x", represented_sigma_mm=1.0)
        assert s.measured_sigma_mm == DEFAULT_SCANNER_ACCURACY_MM

    def test_to_dict_includes_standard_version(self):
        s = build_loa_statement("seg-y", represented_sigma_mm=1.0)
        d = s.to_dict()
        assert d["standard_version"] == "USIBD LOA v3.1"
        assert d["segment_id"] == "seg-y"


class TestSummariseTierDistribution:
    def test_counts_effective_tiers(self):
        statements = [
            build_loa_statement("a", represented_sigma_mm=0.04, measured_sigma_mm=0.04),  # LOA50
            build_loa_statement("b", represented_sigma_mm=0.04, measured_sigma_mm=0.04),  # LOA50
            build_loa_statement("c", represented_sigma_mm=2.0, measured_sigma_mm=0.04),  # LOA30
        ]
        dist = summarise_tier_distribution(statements)
        assert dist["LOA50"] == 2
        assert dist["LOA30"] == 1
        assert dist["LOA40"] == 0
        assert dist["LOA20"] == 0
        assert dist["LOA10"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# 4. LOAStatementModel (Pydantic)
# ═════════════════════════════════════════════════════════════════════════════


class TestLOAStatementModel:
    def test_valid_model(self):
        m = LOAStatementModel(
            segment_id="x",
            measured_sigma_mm=1.0,
            measured_tier="LOA30",
            represented_sigma_mm=2.0,
            represented_tier="LOA30",
            effective_tier="LOA30",
        )
        assert m.standard_version == "USIBD LOA v3.1"

    def test_rejects_unknown_tier(self):
        with pytest.raises(Exception):  # ValidationError
            LOAStatementModel(
                segment_id="x",
                measured_sigma_mm=1.0,
                measured_tier="LOA99",  # invalid
                represented_sigma_mm=1.0,
                represented_tier="LOA30",
                effective_tier="LOA30",
            )

    def test_rejects_negative_sigma(self):
        with pytest.raises(Exception):
            LOAStatementModel(
                segment_id="x",
                measured_sigma_mm=-1.0,
                measured_tier="LOA30",
                represented_sigma_mm=1.0,
                represented_tier="LOA30",
                effective_tier="LOA30",
            )


# ═════════════════════════════════════════════════════════════════════════════
# 5. API endpoint — /sessions/{id}/loa-report
# ═════════════════════════════════════════════════════════════════════════════


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def auth_headers(client):
    async def _get():
        resp = await client.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return _run(_get())


@pytest.fixture
def session_id(client):
    async def _create():
        resp = await client.post(
            "/sessions",
            json={"site_name": "LOA Test Site", "use_synthetic": True},
        )
        return resp.json()["session_id"]

    return _run(_create())


def _seg_with_sigma(sigma_mm: float | None, shape: str = "plane_vertical") -> dict[str, Any]:
    tags = {}
    if sigma_mm is not None:
        tags["loa_sigma_mm"] = sigma_mm
    return {
        "segment_id": str(uuid.uuid4()),
        "zone_id": "zone-A",
        "shape": shape,
        "normal": {"x": 0.0, "y": 1.0, "z": 0.0},
        "centroid": {"x": 1000.0, "y": 500.0, "z": 1500.0},
        "bounding_box": {
            "min_x": 0.0,
            "max_x": 5000.0,
            "min_y": 0.0,
            "max_y": 200.0,
            "min_z": 0.0,
            "max_z": 3000.0,
        },
        "point_count": 5000,
        "confidence": 0.9,
        "tags": tags,
    }


class TestLOAReportEndpoint:
    def test_requires_auth(self, client, session_id):
        resp = _run(client.get(f"/sessions/{session_id}/loa-report"))
        assert resp.status_code == 401

    def test_unknown_session_returns_404(self, client, auth_headers):
        resp = _run(
            client.get(
                "/sessions/does-not-exist/loa-report",
                headers=auth_headers,
            )
        )
        assert resp.status_code == 404

    def test_happy_path_returns_report(self, client, auth_headers, session_id):
        # Ingest a few segments with known sigmas
        payload = {
            "segments": [
                _seg_with_sigma(0.04),  # LOA50
                _seg_with_sigma(2.0),  # LOA30
                _seg_with_sigma(6.0),  # LOA20
            ],
            "scan_metadata": {},
            "source": "revit_plugin",
        }
        r1 = _run(
            client.post(
                f"/sessions/{session_id}/ingest-elements",
                json=payload,
                headers=auth_headers,
            )
        )
        assert r1.status_code == 200

        resp = _run(
            client.get(
                f"/sessions/{session_id}/loa-report",
                headers=auth_headers,
            )
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["standard_version"] == "USIBD LOA v3.1"
        assert body["total_segments"] == 3
        assert body["scanner_accuracy_mm"] == DEFAULT_SCANNER_ACCURACY_MM
        assert "tier_distribution" in body
        assert "worst_tier" in body
        assert len(body["statements"]) == 3

        # Every statement has required keys
        for s in body["statements"]:
            assert {
                "segment_id",
                "measured_tier",
                "represented_tier",
                "effective_tier",
                "measured_sigma_mm",
                "represented_sigma_mm",
                "standard_version",
            } <= set(s.keys())

    def test_segment_without_sigma_defaults_to_loa10(self, client, auth_headers, session_id):
        payload = {
            "segments": [_seg_with_sigma(None)],  # no loa_sigma_mm tag
            "scan_metadata": {},
            "source": "revit_plugin",
        }
        r1 = _run(
            client.post(
                f"/sessions/{session_id}/ingest-elements",
                json=payload,
                headers=auth_headers,
            )
        )
        assert r1.status_code == 200

        resp = _run(
            client.get(
                f"/sessions/{session_id}/loa-report",
                headers=auth_headers,
            )
        )
        body = resp.json()
        assert body["total_segments"] == 1
        # Missing σ → represented_tier defaults to LOA10
        assert body["statements"][0]["represented_tier"] == "LOA10"

    def test_worst_tier_matches_least_accurate_segment(self, client, auth_headers, session_id):
        payload = {
            "segments": [
                _seg_with_sigma(0.04),  # LOA50
                _seg_with_sigma(20.0),  # LOA10 (2σ=40mm)
            ],
            "scan_metadata": {},
            "source": "revit_plugin",
        }
        _run(
            client.post(
                f"/sessions/{session_id}/ingest-elements",
                json=payload,
                headers=auth_headers,
            )
        )
        resp = _run(
            client.get(
                f"/sessions/{session_id}/loa-report",
                headers=auth_headers,
            )
        )
        body = resp.json()
        assert body["worst_tier"] == "LOA10"
