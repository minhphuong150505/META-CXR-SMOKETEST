import json
from pathlib import Path

import pytest

from smoke.runtime import (
    assert_session_eta,
    discover_dataset,
    load_dataset_manifest,
    sha256_file,
)


def test_dataset_discovery_requires_exact_slug_and_manifest(tmp_path):
    target = tmp_path / "private-data"
    target.mkdir()
    (target / "dataset_manifest.json").write_text('{"status":"qa_passed"}')
    assert discover_dataset("owner/private-data", tmp_path) == target


def test_dataset_discovery_rejects_missing_or_ambiguous_manifest(tmp_path):
    (tmp_path / "private-data").mkdir()
    with pytest.raises(RuntimeError, match="found 0"):
        discover_dataset("private-data", tmp_path)


def test_load_manifest_returns_exact_file_hash(tmp_path):
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps({"status": "qa_passed"}), encoding="utf-8")
    payload, resolved, digest = load_dataset_manifest(tmp_path)
    assert payload["status"] == "qa_passed"
    assert resolved == path
    assert digest == sha256_file(path)


def test_eta_guard_reserves_upload_window():
    result = assert_session_eta(2, 20.0, 100, 12.0, 90.0)
    assert result["fits"] is True
    with pytest.raises(RuntimeError, match="exceeds"):
        assert_session_eta(1, 1000.0, 100, 12.0, 90.0)
