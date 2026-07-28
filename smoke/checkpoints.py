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


def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def upload_checkpoint_gcs(bucket_name: str, run_dir: str | Path) -> str:
    """Upload only checkpoint_last.pth, checkpoint_best.pth and log.txt to the
    bucket root (gs://bucket_name/), overwriting the prior session's copies.

    checkpoint_best.pth is optional: a resumed session that never beats the
    carried-forward best produces none locally, and the best already on GCS
    stays authoritative. Each uploaded blob's size is verified against the
    local file to catch a truncated upload without re-downloading the weights.
    """
    run_dir = Path(run_dir)
    required = ("checkpoint_last.pth", "log.txt")
    optional = ("checkpoint_best.pth",)
    bucket = _gcs_client().bucket(bucket_name)
    for name in required + optional:
        f = run_dir / name
        if not f.is_file():
            if name in required:
                raise FileNotFoundError(f"Required upload artifact is missing: {f}")
            continue
        blob = bucket.blob(name)
        blob.upload_from_filename(str(f))
        blob.reload()
        local_size = f.stat().st_size
        if blob.size != local_size:
            raise RuntimeError(
                f"Uploaded {name} size mismatch ({blob.size} != {local_size})"
            )
    return f"gs://{bucket_name}/"


def download_last_checkpoint(bucket_name: str, dest: str | Path):
    """Download only checkpoint_last.pth from the bucket root (gs://bucket_name/)
    into dest.

    Returns the local Path to resume from, or None when no checkpoint exists yet
    (a fresh experiment, so training starts from epoch 0). checkpoint_best.pth is
    never pulled: resuming needs only the last state, and best-tracking is
    restored from fields carried inside checkpoint_last.pth.
    """
    dest = Path(dest)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob("checkpoint_last.pth")
    if not blob.exists():
        return None
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / "checkpoint_last.pth"
    blob.download_to_filename(str(local))
    return local


def download_best_checkpoint(bucket_name: str, dest: str | Path) -> Path:
    """Download checkpoint_best.pth from the bucket root for final evaluation.

    Unlike ``download_last_checkpoint``, a best checkpoint is mandatory for
    held-out sensitivity evaluation, so a missing object is an explicit error.
    """
    dest = Path(dest)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob("checkpoint_best.pth")
    if not blob.exists():
        raise FileNotFoundError(
            f"gs://{bucket_name}/checkpoint_best.pth is missing; finish at least "
            "one validation epoch and upload the Stage-1 checkpoint first"
        )
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / "checkpoint_best.pth"
    blob.download_to_filename(str(local))
    if not local.is_file() or local.stat().st_size == 0:
        raise RuntimeError("Downloaded checkpoint_best.pth is empty")
    return local
