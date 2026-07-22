"""Notebook privacy guard.

Fixtures under ``tests/fixtures/notebooks/`` are entirely synthetic. No content
was copied from a real notebook, and every identifier-shaped number in them was
invented for this test. That is not incidental: copying a real executed
notebook into a fixture in order to test the leak detector would itself commit
the leak.

They are stored as ``*.ipynb.fixture``, not ``*.ipynb``. Several of them are
deliberately dirty -- that is what makes them useful -- and ``--all-tracked``
scans every tracked notebook with no exceptions. Storing them under the
notebook extension would therefore make the repository permanently fail its own
guard. The alternative fixes were both worse: excluding ``tests/`` from
``--all-tracked`` would hide a genuinely leaky notebook parked there later, and
teaching the guard to forgive violations by filename would blunt the detector
itself. Keeping the dirty data out of the notebook namespace leaves the guard
at full strength.

``notebook()`` materialises a fixture as a real ``.ipynb`` under ``tmp_path``,
so the checker still sees an ordinary notebook on disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_notebook_privacy.py"
FIXTURES = Path(__file__).parent / "fixtures" / "notebooks"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_notebook_privacy import (  # noqa: E402
    EXIT_ERROR,
    EXIT_OK,
    EXIT_VIOLATION,
    check_notebook,
    main,
    redact,
    sanitize_notebook,
)


def fixture_text(name: str) -> str:
    return (FIXTURES / f"{name}.ipynb.fixture").read_text(encoding="utf-8")


def notebook(tmp_path: Path, name: str) -> Path:
    """Write fixture ``name`` out as a real notebook under ``tmp_path``."""
    destination = tmp_path / f"{name}.ipynb"
    destination.write_text(fixture_text(name), encoding="utf-8")
    return destination


def kinds(path: Path) -> set[str]:
    return {violation.kind for violation in check_notebook(path)}


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_clean_notebook_is_accepted(tmp_path):
    clean = notebook(tmp_path, "clean")
    assert check_notebook(clean) == []
    assert main(["--path", str(clean)]) == EXIT_OK


def test_executed_output_is_detected(tmp_path):
    assert "executed_output_present" in kinds(notebook(tmp_path, "executed_output"))


def test_execution_count_is_detected(tmp_path):
    assert "execution_count_set" in kinds(notebook(tmp_path, "executed_output"))


def test_synthetic_identifier_keys_are_detected(tmp_path):
    found = kinds(notebook(tmp_path, "synthetic_identifier"))
    assert "identifier_key:subject_id" in found
    assert "identifier_key:study_id" in found


def test_identifier_values_are_detected_with_context(tmp_path):
    found = kinds(notebook(tmp_path, "synthetic_identifier"))
    assert "identifier_value_with_context" in found


def test_mimic_data_path_is_detected(tmp_path):
    assert "mimic_data_path" in kinds(notebook(tmp_path, "synthetic_identifier"))


def test_credentials_are_detected(tmp_path):
    found = kinds(notebook(tmp_path, "credential_like"))
    assert "hf_token" in found
    assert "gcs_signed_url" in found


def test_credentials_in_source_are_detected_even_without_outputs(tmp_path):
    """A token pasted into a cell is a leak whether or not the cell was run."""
    violations = check_notebook(notebook(tmp_path, "credential_like"))
    assert any(v.location == "source" and v.kind == "hf_token" for v in violations)


def test_bare_kaggle_ids_are_not_flagged(tmp_path):
    """Tracked notebooks carry legitimate 8-digit Kaggle datasetId/sourceId.

    Flagging those would produce constant false positives and train people to
    ignore the checker, which is worse than not having one.
    """
    found = kinds(notebook(tmp_path, "kaggle_ids"))
    assert "identifier_value_with_context" not in found
    assert found == set()


def test_eight_digit_number_without_mimic_context_is_ignored(tmp_path):
    target = tmp_path / "n.ipynb"
    target.write_text(json.dumps({
        "cells": [{"cell_type": "code", "source": ["x = 10478126\n"],
                   "outputs": [], "execution_count": None, "metadata": {}}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }), encoding="utf-8")
    assert check_notebook(target) == []


def test_unreadable_notebook_is_reported_not_crashed(tmp_path):
    broken = tmp_path / "broken.ipynb"
    broken.write_text("{not json", encoding="utf-8")
    assert "unreadable_notebook" in kinds(broken)


# --------------------------------------------------------------------------
# redaction -- the checker must not leak while reporting a leak
# --------------------------------------------------------------------------


def test_redact_masks_the_middle():
    assert redact("10000032") == "10****32"


def test_redact_hides_short_values_entirely():
    assert redact("abcd") == "****"


def test_no_full_identifier_appears_in_any_violation(tmp_path):
    """The values in the fixture must never be reproduced in full."""
    violations = check_notebook(notebook(tmp_path, "synthetic_identifier"))
    rendered = " ".join(v.format() for v in violations)
    for value in ("19999901", "59999901", "19999902", "59999902"):
        assert value not in rendered


def test_no_full_credential_appears_in_any_violation(tmp_path):
    violations = check_notebook(notebook(tmp_path, "credential_like"))
    rendered = " ".join(v.format() for v in violations)
    assert "hf_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in rendered
    assert "AAAAsignatureAAAA" not in rendered


def test_reported_output_is_redacted(tmp_path, capsys):
    main(["--path", str(notebook(tmp_path, "synthetic_identifier"))])
    captured = capsys.readouterr()
    assert "19999901" not in captured.err + captured.out
    assert "NOTEBOOK PRIVACY CHECK FAILED" in captured.err


# --------------------------------------------------------------------------
# sanitize
# --------------------------------------------------------------------------


def test_sanitize_strips_outputs_and_counts(tmp_path):
    destination = tmp_path / "sanitized.ipynb"
    sanitize_notebook(notebook(tmp_path, "synthetic_identifier"), destination)
    assert check_notebook(destination) == []


def test_sanitize_preserves_cell_source(tmp_path):
    source_nb = notebook(tmp_path, "synthetic_identifier")
    destination = tmp_path / "sanitized.ipynb"
    sanitize_notebook(source_nb, destination)

    before = json.loads(source_nb.read_text(encoding="utf-8"))["cells"]
    after = json.loads(destination.read_text(encoding="utf-8"))["cells"]
    assert [c["source"] for c in after] == [c["source"] for c in before]


def test_sanitize_does_not_modify_the_input(tmp_path):
    source_nb = notebook(tmp_path, "synthetic_identifier")
    original = source_nb.read_text(encoding="utf-8")
    sanitize_notebook(source_nb, tmp_path / "out.ipynb")
    assert source_nb.read_text(encoding="utf-8") == original


def test_sanitize_without_output_is_refused(tmp_path):
    target = tmp_path / "n.ipynb"
    target.write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4,
                                  "nbformat_minor": 5}), encoding="utf-8")
    assert main(["--sanitize", str(target)]) == EXIT_ERROR


def test_sanitize_in_place_allowed_for_untracked_file(tmp_path):
    target = notebook(tmp_path, "executed_output")
    assert main(["--sanitize", str(target), "--overwrite-in-place"]) == EXIT_OK
    assert check_notebook(target) == []


def test_sanitize_refuses_to_overwrite_a_tracked_notebook(capsys):
    """Editing a tracked notebook in place would rewrite history-bound content."""
    tracked = REPO_ROOT / "notebooks" / "01_generate_mimic_cxr_cleaned_csv.ipynb"
    if not tracked.is_file():
        pytest.skip("tracked notebook fixture is absent in this checkout")
    assert main(["--sanitize", str(tracked), "--overwrite-in-place"]) == EXIT_ERROR
    assert "refusing to overwrite" in capsys.readouterr().err


def test_missing_notebook_is_an_error(tmp_path):
    assert main(["--path", str(tmp_path / "nope.ipynb")]) == EXIT_ERROR


# --------------------------------------------------------------------------
# git integration
# --------------------------------------------------------------------------


def test_staged_mode_checks_only_staged_files(tmp_path):
    """A dirty-but-unstaged notebook must not block an unrelated commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)

    (repo / "dirty.ipynb").write_text(fixture_text("synthetic_identifier"), encoding="utf-8")
    (repo / "note.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "note.txt"], cwd=repo, check=True)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--staged"], cwd=repo, capture_output=True, text=True
    )
    assert result.returncode == EXIT_OK

    subprocess.run(["git", "add", "dirty.ipynb"], cwd=repo, check=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--staged"], cwd=repo, capture_output=True, text=True
    )
    assert result.returncode == EXIT_VIOLATION
    assert "19999901" not in result.stdout + result.stderr


def test_the_private_notebook_is_still_ignored_and_untracked():
    """Regression guard for the notebook known to hold real MIMIC identifiers.

    Deliberately does not open or read the file.
    """
    target = "preporcessing/kltn-data-preprocessing.ipynb"
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", target], cwd=REPO_ROOT, check=False
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", target],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    assert ignored.returncode == 0, f"{target} is no longer git-ignored"
    assert tracked.returncode != 0, f"{target} has become tracked by Git"


def test_all_tracked_notebooks_pass_the_guard():
    """The notebooks currently in the repo must be clean."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--all-tracked"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == EXIT_OK, result.stderr
