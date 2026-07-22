from __future__ import annotations

import json
from pathlib import Path

from smoke.runtime import sha256_file


CHECKPOINT_FILES = ("checkpoint_best.pth", "checkpoint_last.pth")


def write_artifact_manifest(run_dir: str | Path, identity: dict) -> Path:
    run_dir = Path(run_dir)
    files = {}
    for name in CHECKPOINT_FILES:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint is missing: {path}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    payload = {"identity": identity, "files": files}
    path = run_dir / "artifact_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def assert_private_kaggle_dataset(handle: str) -> None:
    """Fail closed unless the pre-created checkpoint dataset is private."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dataset = api.dataset_view(handle)
    is_private = getattr(dataset, "isPrivate", None)
    if is_private is None:
        is_private = getattr(dataset, "is_private", None)
    if is_private is not True:
        raise RuntimeError(
            f"Checkpoint dataset {handle!r} is not verifiably private; upload refused"
        )


def upload_private_checkpoint_dataset(handle: str, run_dir: str | Path) -> None:
    """Upload only after privacy and local manifest checks pass."""
    import kagglehub

    assert_private_kaggle_dataset(handle)
    run_dir = Path(run_dir)
    manifest = run_dir / "artifact_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("artifact_manifest.json must be written before upload")
    kagglehub.dataset_upload(
        handle,
        str(run_dir),
        version_notes="META-CXR private checkpoint/resume artifact",
        ignore_patterns=["*.tmp", "predictions/", "ground_truths/"],
    )

    # Verify a fresh authenticated read of the small manifest. Checkpoints are
    # retained locally; no destructive cleanup occurs even after success.
    downloaded = Path(
        kagglehub.dataset_download(
            handle, path="artifact_manifest.json", force_download=True
        )
    )
    if downloaded.is_dir():
        downloaded = downloaded / "artifact_manifest.json"
    if sha256_file(downloaded) != sha256_file(manifest):
        raise RuntimeError("Uploaded artifact manifest failed SHA-256 verification")


def upload_private_results_dataset(handle: str, result_dir: str | Path) -> None:
    """Publish aggregate-only evaluation results to a pre-created private dataset."""
    import kagglehub

    assert_private_kaggle_dataset(handle)
    result_dir = Path(result_dir)
    result = result_dir / "encoder_sensitivity.json"
    if not result.is_file():
        raise FileNotFoundError(result)
    manifest = result_dir / "result_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "encoder_sensitivity.json": sha256_file(result),
                "encoder_sensitivity.md": sha256_file(
                    result_dir / "encoder_sensitivity.md"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    kagglehub.dataset_upload(
        handle,
        str(result_dir),
        version_notes="META-CXR aggregate encoder sensitivity metrics",
    )
    downloaded = Path(
        kagglehub.dataset_download(handle, path="result_manifest.json", force_download=True)
    )
    if downloaded.is_dir():
        downloaded = downloaded / "result_manifest.json"
    if sha256_file(downloaded) != sha256_file(manifest):
        raise RuntimeError("Uploaded result manifest failed SHA-256 verification")
