"""Tests for processor_cli.py — the Revit sidecar CLI.

All tests use synthetic point data (no real scan file required).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so the agent package is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.models import (
    BoundingBox,
    ElementType,
    GeometrySegment,
    Point3D,
    SegmentShape,
)
from agent.tools.processor_cli import _parse_args, main, process

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_segment(idx: int = 0) -> GeometrySegment:
    return GeometrySegment(
        zone_id="zone-001",
        shape=SegmentShape.PLANE_VERTICAL,
        centroid=Point3D(x=float(idx * 1000), y=0.0, z=1500.0),
        bounding_box=BoundingBox(
            min_x=0.0,
            max_x=5000.0,
            min_y=0.0,
            max_y=200.0,
            min_z=0.0,
            max_z=3000.0,
        ),
        point_count=5000 + idx,
        confidence=0.92,
    )


def _fake_scan_info(path: Path, warnings: list[str] | None = None):
    from agent.tools.scan_tools import ScanFileInfo

    return ScanFileInfo(
        path=path,
        format="e57",
        size_bytes=1_000_000,
        size_mb=1.0,
        valid=True,
        point_count=50_000,
        warnings=warnings or [],
    )


# ── Argument parsing ──────────────────────────────────────────────────────────


class TestArgParsing:
    def test_required_input(self):
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_defaults(self, tmp_path):
        f = tmp_path / "scan.e57"
        f.touch()
        args = _parse_args(["--input", str(f)])
        assert args.zone == "zone-001"
        assert args.voxel_size == 0.01
        assert args.output is None

    def test_all_flags(self, tmp_path):
        f = tmp_path / "scan.las"
        f.touch()
        out = tmp_path / "results.json"
        args = _parse_args(
            [
                "--input",
                str(f),
                "--zone",
                "zone-003",
                "--voxel-size",
                "0.025",
                "--output",
                str(out),
            ]
        )
        assert args.zone == "zone-003"
        assert args.voxel_size == 0.025
        assert args.output == out

    def test_short_flags(self, tmp_path):
        f = tmp_path / "scan.ply"
        f.touch()
        out = tmp_path / "out.json"
        args = _parse_args(["-i", str(f), "-z", "zone-X", "-o", str(out)])
        assert args.zone == "zone-X"
        assert args.output == out

    def test_semantic_flags(self, tmp_path):
        f = tmp_path / "scan.e57"
        f.touch()
        preds = tmp_path / "preds.npy"
        preds.write_bytes(b"x")
        args = _parse_args(
            [
                "--input",
                str(f),
                "--semantic-enable",
                "--semantic-model",
                "ptv3",
                "--semantic-checkpoint",
                str(preds),
                "--semantic-wall-threshold",
                "0.66",
                "--semantic-no-fallback",
            ]
        )
        assert args.semantic_enable is True
        assert args.semantic_model == "ptv3"
        assert args.semantic_checkpoint == preds
        assert args.semantic_wall_threshold == pytest.approx(0.66)
        assert args.semantic_no_fallback is True


# ── process() — happy path ────────────────────────────────────────────────────


class TestProcessHappyPath:
    def test_writes_json_file(self, tmp_path):
        scan = tmp_path / "test.e57"
        scan.touch()
        out = tmp_path / "results.json"
        segs = [_make_segment(i) for i in range(3)]

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=segs),
        ):
            rc = process(scan, "zone-001", 0.01, out)

        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["schema_version"] == "1.0"
        assert len(data["segments"]) == 3

    def test_metadata_fields_present(self, tmp_path):
        scan = tmp_path / "scan.las"
        scan.write_bytes(b"\x00" * 100)
        out = tmp_path / "results.json"
        segs = [_make_segment()]

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=segs),
        ):
            process(scan, "zone-002", 0.02, out)

        meta = json.loads(out.read_text())["metadata"]
        for key in (
            "processor_version",
            "processed_at",
            "input_file",
            "zone_id",
            "voxel_size_m",
            "segment_count",
            "processing_time_s",
        ):
            assert key in meta, f"Missing metadata key: {key}"

    def test_semantic_metadata_present_when_enabled(self, tmp_path):
        scan = tmp_path / "scan.e57"
        scan.touch()
        out = tmp_path / "results.json"
        segs = [_make_segment()]

        with (
            patch("agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)),
            patch("agent.tools.processor_cli.run_segmentation", return_value=segs),
            patch(
                "agent.tools.processor_cli.get_last_semantic_report",
                return_value={
                    "enabled": True,
                    "model": "ptv3",
                    "success": True,
                    "provider": "file",
                    "wall_points": 123,
                    "total_points": 456,
                },
            ),
        ):
            process(
                scan,
                "zone-001",
                0.01,
                out,
                semantic_enabled=True,
                semantic_model="ptv3",
                semantic_wall_threshold=0.55,
            )

        meta = json.loads(out.read_text())["metadata"]
        assert meta["semantic_model"] == "ptv3"
        assert meta["semantic_wall_threshold"] == pytest.approx(0.55)
        assert meta["semantic"]["enabled"] is True

    def test_segment_fields_serialised(self, tmp_path):
        scan = tmp_path / "scan.ply"
        scan.touch()
        out = tmp_path / "results.json"
        seg = _make_segment(0)

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=[seg]),
        ):
            process(scan, "zone-001", 0.01, out)

        segments = json.loads(out.read_text())["segments"]
        assert len(segments) == 1
        s = segments[0]
        assert "segment_id" in s
        assert s["zone_id"] == "zone-001"
        assert s["shape"] == "plane_vertical"
        assert s["confidence"] == pytest.approx(0.92)
        assert s["point_count"] == 5000
        assert "centroid" in s
        assert "bounding_box" in s

    def test_stdout_when_no_output_path(self, tmp_path, capsys):
        scan = tmp_path / "scan.xyz"
        scan.touch()
        segs = [_make_segment()]

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=segs),
        ):
            rc = process(scan, "zone-001", 0.01, None)

        assert rc == 0
        captured = capsys.readouterr().out
        # Should contain valid JSON somewhere in stdout
        json_lines = [l for l in captured.splitlines() if l.startswith("{")]
        assert json_lines, "Expected JSON output on stdout"

    def test_unknown_point_count_prints_unknown(self, tmp_path, capsys):
        scan = tmp_path / "scan.xyz"
        scan.touch()
        out = tmp_path / "results.json"
        info = _fake_scan_info(scan)
        info.point_count = None

        with (
            patch("agent.tools.processor_cli.validate_scan_file", return_value=info),
            patch("agent.tools.processor_cli.run_segmentation", return_value=[_make_segment()]),
        ):
            process(scan, "zone-001", 0.01, out)

        captured = capsys.readouterr().out
        assert "from unknown points" in captured

    def test_filters_implausible_floor_level_cable_trays(self, tmp_path):
        scan = tmp_path / "scan.e57"
        scan.touch()
        out = tmp_path / "results.json"
        seg = GeometrySegment(
            zone_id="zone-001",
            shape=SegmentShape.BOX,
            centroid=Point3D(x=0.0, y=600.0, z=0.0),
            bounding_box=BoundingBox(
                min_x=-200.0,
                max_x=200.0,
                min_y=0.0,
                max_y=1200.0,
                min_z=-100.0,
                max_z=100.0,
            ),
            point_count=2000,
            confidence=0.95,
            element_type=ElementType.CABLE_TRAY,
        )

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=[seg]),
        ):
            process(scan, "zone-001", 0.01, out)

        payload = json.loads(out.read_text())
        assert all(s.get("element_type") != "cable_tray" for s in payload["segments"])
        assert any("implausible_cable_tray" in warning for warning in payload["warnings"])

    def test_zero_point_segments_message_for_dict_segments(self, tmp_path, capsys):
        scan = tmp_path / "scan.las"
        scan.touch()
        out = tmp_path / "results.json"
        info = _fake_scan_info(scan)
        info.point_count = 0
        segs = [
            {
                "segment_id": "seg-0",
                "zone_id": "zone-001",
                "shape": "plane_vertical",
                "confidence": 0.5,
                "point_count": 0,
            }
        ]

        with (
            patch("agent.tools.processor_cli.validate_scan_file", return_value=info),
            patch("agent.tools.processor_cli.run_segmentation", return_value=segs),
        ):
            process(scan, "zone-001", 0.01, out)

        captured = capsys.readouterr().out
        assert "filtered 1 segments from 0 points" in captured

    def test_warnings_from_scan_info_included(self, tmp_path):
        scan = tmp_path / "scan.e57"
        scan.touch()
        out = tmp_path / "results.json"

        info = _fake_scan_info(scan, warnings=["Low point density in sector A"])

        with (
            patch("agent.tools.processor_cli.validate_scan_file", return_value=info),
            patch("agent.tools.processor_cli.run_segmentation", return_value=[]),
        ):
            process(scan, "zone-001", 0.01, out)

        data = json.loads(out.read_text())
        assert any("density" in w for w in data["warnings"])

    def test_zero_segments_is_valid(self, tmp_path):
        scan = tmp_path / "empty.las"
        scan.touch()
        out = tmp_path / "results.json"

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=[]),
        ):
            rc = process(scan, "zone-001", 0.01, out)

        assert rc == 0
        data = json.loads(out.read_text())
        assert data["segments"] == []


# ── process() — error paths ───────────────────────────────────────────────────


class TestProcessErrors:
    def test_missing_file_returns_exit_1(self, tmp_path):
        missing = tmp_path / "ghost.e57"
        out = tmp_path / "results.json"

        with patch(
            "agent.tools.processor_cli.validate_scan_file",
            side_effect=FileNotFoundError("File not found"),
        ):
            rc = process(missing, "zone-001", 0.01, out)

        assert rc == 1

    def test_bad_format_returns_exit_1(self, tmp_path):
        bad = tmp_path / "scan.txt"
        bad.touch()
        out = tmp_path / "results.json"

        with patch(
            "agent.tools.processor_cli.validate_scan_file",
            side_effect=ValueError("Unsupported format"),
        ):
            rc = process(bad, "zone-001", 0.01, out)

        assert rc == 1

    def test_segmentation_error_returns_exit_1(self, tmp_path):
        scan = tmp_path / "scan.e57"
        scan.touch()
        out = tmp_path / "results.json"

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch(
                "agent.tools.processor_cli.run_segmentation",
                side_effect=RuntimeError("Out of memory"),
            ),
        ):
            rc = process(scan, "zone-001", 0.01, out)

        assert rc == 1

    def test_error_writes_fatal_warning_to_json(self, tmp_path):
        scan = tmp_path / "scan.e57"
        out = tmp_path / "results.json"

        with patch(
            "agent.tools.processor_cli.validate_scan_file",
            side_effect=ValueError("Bad magic bytes"),
        ):
            process(scan, "zone-001", 0.01, out)

        data = json.loads(out.read_text())
        assert any("FATAL" in w for w in data["warnings"])
        assert data["segments"] == []


# ── main() entrypoint ─────────────────────────────────────────────────────────


class TestMainEntrypoint:
    def test_main_returns_0_on_success(self, tmp_path):
        scan = tmp_path / "scan.e57"
        scan.touch()
        out = tmp_path / "results.json"

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=[_make_segment()]),
        ):
            rc = main(["--input", str(scan), "--output", str(out)])

        assert rc == 0

    def test_main_returns_1_on_error(self, tmp_path):
        scan = tmp_path / "scan.e57"
        out = tmp_path / "results.json"

        with patch(
            "agent.tools.processor_cli.validate_scan_file",
            side_effect=FileNotFoundError("not found"),
        ):
            rc = main(["--input", str(scan), "--output", str(out)])

        assert rc == 1

    def test_main_returns_2_on_bad_args(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2


# ── Progress tokens emitted ───────────────────────────────────────────────────


class TestProgressOutput:
    def test_done_token_emitted(self, tmp_path, capsys):
        scan = tmp_path / "scan.e57"
        scan.touch()
        out = tmp_path / "results.json"

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=[]),
        ):
            process(scan, "zone-001", 0.01, out)

        stdout = capsys.readouterr().out
        assert "PROGRESS:100:Done" in stdout

    def test_validate_token_emitted(self, tmp_path, capsys):
        scan = tmp_path / "scan.e57"
        scan.touch()
        out = tmp_path / "results.json"

        with (
            patch(
                "agent.tools.processor_cli.validate_scan_file", return_value=_fake_scan_info(scan)
            ),
            patch("agent.tools.processor_cli.run_segmentation", return_value=[]),
        ):
            process(scan, "zone-001", 0.01, out)

        stdout = capsys.readouterr().out
        assert "PROGRESS:10:" in stdout
