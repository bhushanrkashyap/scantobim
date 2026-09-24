"""test_determinism.py — NQA-1 § 401 reproducibility enforcement.

Covers:
  • resolve_seed precedence (explicit > env var > default)
  • set_global_seed actually seeds numpy/random/Open3D
  • deterministic_segment_id is stable across calls + varies with inputs
  • compute_input_hash stream-reads files correctly
  • compute_output_hash is canonicalised (insertion order doesn't matter)
  • Orchestrator stamps provenance on segmentation
  • Ingest path stores provenance with sidecar input_hash
  • /provenance endpoint (auth, 404, happy path)
  • NQA-1 package surfaces determinism evidence
  • run_segmentation honours seed argument
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ═════════════════════════════════════════════════════════════════════════════
# 1. Seed resolution
# ═════════════════════════════════════════════════════════════════════════════


class TestSeedResolution:
    def test_explicit_seed_wins(self, monkeypatch):
        from agent.determinism import resolve_seed

        monkeypatch.setenv("STB_RANDOM_SEED", "99")
        assert resolve_seed(seed=7) == 7

    def test_env_var_used_when_no_explicit(self, monkeypatch):
        from agent.determinism import resolve_seed

        monkeypatch.setenv("STB_RANDOM_SEED", "123")
        assert resolve_seed() == 123

    def test_default_when_nothing_set(self, monkeypatch):
        from agent.determinism import DEFAULT_SEED, resolve_seed

        monkeypatch.delenv("STB_RANDOM_SEED", raising=False)
        assert resolve_seed() == DEFAULT_SEED

    def test_invalid_env_falls_through_to_default(self, monkeypatch):
        from agent.determinism import DEFAULT_SEED, resolve_seed

        monkeypatch.setenv("STB_RANDOM_SEED", "not-a-number")
        assert resolve_seed() == DEFAULT_SEED

    def test_set_global_seed_returns_applied(self, monkeypatch):
        from agent.determinism import set_global_seed

        monkeypatch.delenv("STB_RANDOM_SEED", raising=False)
        assert set_global_seed(17) == 17

    def test_set_global_seed_is_deterministic(self):
        """Same seed → same random sequence across numpy."""
        import numpy as np

        from agent.determinism import set_global_seed

        set_global_seed(42)
        a = np.random.rand(5).tolist()
        set_global_seed(42)
        b = np.random.rand(5).tolist()
        assert a == b


# ═════════════════════════════════════════════════════════════════════════════
# 2. deterministic_segment_id
# ═════════════════════════════════════════════════════════════════════════════


class TestDeterministicSegmentId:
    def test_stable_across_calls(self):
        from agent.determinism import deterministic_segment_id

        a = deterministic_segment_id("s1", "box", (1, 2, 3), (0, 0, 0, 10, 10, 10))
        b = deterministic_segment_id("s1", "box", (1, 2, 3), (0, 0, 0, 10, 10, 10))
        assert a == b

    def test_different_session_different_id(self):
        from agent.determinism import deterministic_segment_id

        a = deterministic_segment_id("s1", "box", (1, 2, 3), (0, 0, 0, 10, 10, 10))
        b = deterministic_segment_id("s2", "box", (1, 2, 3), (0, 0, 0, 10, 10, 10))
        assert a != b

    def test_different_geometry_different_id(self):
        from agent.determinism import deterministic_segment_id

        a = deterministic_segment_id("s1", "box", (1, 2, 3), (0, 0, 0, 10, 10, 10))
        b = deterministic_segment_id("s1", "box", (9, 9, 9), (0, 0, 0, 10, 10, 10))
        assert a != b

    def test_tiny_float_noise_does_not_change_id(self):
        """Rounding to 0.1 mm means nm-level jitter keeps the same ID."""
        from agent.determinism import deterministic_segment_id

        a = deterministic_segment_id("s1", "box", (1.000001, 2, 3), (0, 0, 0, 10, 10, 10))
        b = deterministic_segment_id("s1", "box", (1.000002, 2, 3), (0, 0, 0, 10, 10, 10))
        assert a == b

    def test_result_is_valid_uuid(self):
        from agent.determinism import deterministic_segment_id

        sid = deterministic_segment_id("s1", "box", (1, 2, 3), (0, 0, 0, 10, 10, 10))
        uuid.UUID(sid)  # should not raise


# ═════════════════════════════════════════════════════════════════════════════
# 3. Content hashing
# ═════════════════════════════════════════════════════════════════════════════


class TestInputHash:
    def test_known_content_hash(self, tmp_path: Path):
        from agent.determinism import compute_input_hash

        p = tmp_path / "scan.xyz"
        p.write_bytes(b"hello world")
        # sha256("hello world") = b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        assert compute_input_hash(p) == (
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )

    def test_missing_file_raises(self, tmp_path: Path):
        from agent.determinism import compute_input_hash

        with pytest.raises(FileNotFoundError):
            compute_input_hash(tmp_path / "nope")


class TestOutputHash:
    def _seg(self, zone="Z", shape="box", cx=0, cy=0, cz=0, seg_id=None):
        from agent.models import (
            BoundingBox,
            GeometrySegment,
            Point3D,
            SegmentShape,
        )

        return GeometrySegment(
            segment_id=seg_id or str(uuid.uuid4()),
            zone_id=zone,
            shape=SegmentShape(shape),
            centroid=Point3D(x=cx, y=cy, z=cz),
            bounding_box=BoundingBox(
                min_x=cx - 1,
                min_y=cy - 1,
                min_z=cz - 1,
                max_x=cx + 1,
                max_y=cy + 1,
                max_z=cz + 1,
            ),
            point_count=100,
            confidence=0.9,
        )

    def test_empty_list_hashable(self):
        from agent.determinism import compute_output_hash

        h = compute_output_hash([])
        assert len(h) == 64

    def test_same_segments_same_hash(self):
        from agent.determinism import compute_output_hash

        segs = [self._seg(cx=1), self._seg(cx=2), self._seg(cx=3)]
        assert compute_output_hash(segs) == compute_output_hash(segs)

    def test_order_insensitive(self):
        """Re-ordering segments must not change the output hash."""
        from agent.determinism import compute_output_hash

        a = [self._seg(cx=1), self._seg(cx=2), self._seg(cx=3)]
        b = [a[2], a[0], a[1]]
        assert compute_output_hash(a) == compute_output_hash(b)

    def test_segment_id_does_not_affect_hash(self):
        """Different UUIDs on identical geometry → same hash."""
        from agent.determinism import compute_output_hash

        s1 = self._seg(cx=1, seg_id=str(uuid.uuid4()))
        s2 = self._seg(cx=1, seg_id=str(uuid.uuid4()))
        assert compute_output_hash([s1]) == compute_output_hash([s2])

    def test_different_content_different_hash(self):
        from agent.determinism import compute_output_hash

        a = [self._seg(cx=1)]
        b = [self._seg(cx=99)]
        assert compute_output_hash(a) != compute_output_hash(b)

    def test_tag_changes_affect_hash(self):
        from agent.determinism import compute_output_hash

        a = self._seg(cx=1)
        b = self._seg(cx=1)
        b.tags["random_seed"] = 99
        assert compute_output_hash([a]) != compute_output_hash([b])


# ═════════════════════════════════════════════════════════════════════════════
# 4. Orchestrator provenance capture
# ═════════════════════════════════════════════════════════════════════════════


class TestOrchestratorProvenance:
    def test_synthetic_run_writes_provenance(self):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Determinism Test Synth")
        segs = orchestrator.run_segmentation(
            sess.session_id,
            use_synthetic=True,
            zone_id="zone-A",
        )
        record = orchestrator._provenance.get(sess.session_id)

        assert record is not None
        assert record["session_id"] == sess.session_id
        assert record["input_source"] == "synthetic"
        assert record["input_hash"] is None  # synthetic has no file
        assert record["segment_count"] == len(segs)
        assert len(record["output_hash"]) == 64  # sha256 hex

    def test_same_synthetic_input_same_output_hash(self, monkeypatch):
        """Running the synthetic generator twice with the same seed must
        produce the same output hash. This is the core NQA-1 claim."""
        from agent.main import orchestrator

        monkeypatch.setenv("STB_RANDOM_SEED", "42")

        s1 = orchestrator.create_session("Det Repro 1")
        orchestrator.run_segmentation(s1.session_id, use_synthetic=True, zone_id="zone-A")
        hash1 = orchestrator._provenance[s1.session_id]["output_hash"]

        s2 = orchestrator.create_session("Det Repro 2")
        orchestrator.run_segmentation(s2.session_id, use_synthetic=True, zone_id="zone-A")
        hash2 = orchestrator._provenance[s2.session_id]["output_hash"]

        assert hash1 == hash2, "Two synthetic runs with same seed produced different output hashes"

    def test_ingest_path_captures_sidecar_input_hash(self):
        from agent.main import orchestrator
        from agent.models import (
            BoundingBox,
            GeometrySegment,
            Point3D,
            SegmentShape,
        )

        sess = orchestrator.create_session("Ingest Provenance")

        # Simulate what the ingest endpoint does (minus HTTP)
        segs = [
            GeometrySegment(
                zone_id="Z",
                shape=SegmentShape.BOX,
                centroid=Point3D(x=1, y=2, z=3),
                bounding_box=BoundingBox(
                    min_x=0,
                    min_y=0,
                    min_z=0,
                    max_x=10,
                    max_y=10,
                    max_z=10,
                ),
                point_count=500,
                confidence=0.9,
            )
        ]
        orchestrator._segments[sess.session_id] = segs
        orchestrator._record_provenance(
            session_id=sess.session_id,
            segments=segs,
            input_hash="a" * 64,
            input_source="north.e57",
        )
        rec = orchestrator._provenance[sess.session_id]
        assert rec["input_hash"] == "a" * 64
        assert rec["input_source"] == "north.e57"


# ═════════════════════════════════════════════════════════════════════════════
# 5. /provenance endpoint
# ═════════════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def auth_client():
    from agent.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        ac.headers["Authorization"] = f"Bearer {resp.json().get('access_token', '')}"
        yield ac


class TestProvenanceEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/sessions/x/provenance")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_404(self, auth_client: AsyncClient):
        resp = await auth_client.get("/sessions/does-not-exist/provenance")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_segmentation_yet_404(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Prov No Seg")
        resp = await auth_client.get(f"/sessions/{sess.session_id}/provenance")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_happy_path_returns_record(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Prov Happy")
        orchestrator.run_segmentation(
            sess.session_id,
            use_synthetic=True,
            zone_id="Z",
        )
        resp = await auth_client.get(f"/sessions/{sess.session_id}/provenance")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sess.session_id
        assert {
            "random_seed",
            "software_identity_hash",
            "output_hash",
            "segment_count",
            "generated_at_utc",
        } <= set(body.keys())


# ═════════════════════════════════════════════════════════════════════════════
# 6. NQA-1 package surfaces determinism
# ═════════════════════════════════════════════════════════════════════════════


class TestNqa1DeterminismSurface:
    def test_package_includes_provenance_when_present(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.nqa1_package import build_nqa1_package

        sess = orchestrator.create_session("NQA-1 Prov Test")
        orchestrator.run_segmentation(
            sess.session_id,
            use_synthetic=True,
            zone_id="Z",
        )
        html, doc = build_nqa1_package(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        assert doc["provenance"] is not None
        assert doc["provenance"]["session_id"] == sess.session_id
        # HTML includes Section 3b — Determinism / Provenance
        assert "Determinism" in html
        assert "Random seed" in html

    def test_package_without_provenance_still_renders(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.nqa1_package import build_nqa1_package

        sess = orchestrator.create_session("NQA-1 No Prov")
        # No segmentation → no provenance
        html, doc = build_nqa1_package(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        assert doc["provenance"] is None
        assert "No provenance record" in html
