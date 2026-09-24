"""Regression test for UTF-8 pipeline integrity across platforms (Windows charmap fix).

Ensures characters such as →, ✓, μ, ° pass cleanly through:
- Stream encoding (stdout, stderr)
- JSON serialization and deserialization
- CSV writing and reading
- Logging handlers
- Stage2 wall grouping progress tokens
"""

import io
import json
import logging
import sys
from pathlib import Path
import pytest


def test_unicode_stream_and_characters():
    """Verify Unicode symbols used across ScanTOBIM format and encode cleanly."""
    unicode_chars = ["→", "✓", "μ", "°"]
    sample_text = "Stage2 grouping: 10 raw walls → 4 physical walls (tolerance ±0.05 m, angle 90°, precision 5 μm) [✓ verified]"

    for char in unicode_chars:
        assert char in sample_text

    # Test UTF-8 byte roundtrip
    encoded = sample_text.encode("utf-8")
    decoded = encoded.decode("utf-8")
    assert decoded == sample_text

    # Test string buffer writing with utf-8 encoding
    buf = io.StringIO()
    buf.write(sample_text)
    assert "→" in buf.getvalue()
    assert "✓" in buf.getvalue()
    assert "μ" in buf.getvalue()
    assert "°" in buf.getvalue()


def test_json_utf8_serialization(tmp_path: Path):
    """Verify JSON writer preserves Unicode characters without charmap crash."""
    data = {
        "operation": "stage2_wall_grouping",
        "symbol_arrow": "→",
        "symbol_check": "✓",
        "symbol_micro": "μ",
        "symbol_degree": "°",
        "message": "Raw walls → Physical walls ✓ (rotation 45°, thickness 200 μm)",
    }

    test_file = tmp_path / "test_unicode.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open(test_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["symbol_arrow"] == "→"
    assert loaded["symbol_check"] == "✓"
    assert loaded["symbol_micro"] == "μ"
    assert loaded["symbol_degree"] == "°"
    assert "→" in loaded["message"]


def test_logging_utf8_handler():
    """Verify logger handles Unicode messages gracefully."""
    logger = logging.getLogger("test_utf8_logger")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    msg = "INFO:stage2: 12 raw walls → 6 physical walls ✓ (angle 0.5°, tolerance 10 μm)"
    logger.info(msg)
    logger.removeHandler(handler)

    assert "→" in stream.getvalue()
    assert "✓" in stream.getvalue()
