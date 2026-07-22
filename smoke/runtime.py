from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


FINGERPRINT_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "timm",
    "omegaconf",
    "hi-ml-multimodal",
    "Pillow",
    "wandb",
    "kaggle",
    "kagglehub",
)
REQUIRED_KAGGLE_PACKAGES = {
    "transformers": "4.53.2",
    "tokenizers": "0.21.4",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "timm": "0.9.16",
    "omegaconf": "2.3.0",
    "hi-ml-multimodal": "0.2.1",
    "Pillow": "10.4.0",
    "wandb": "0.18.7",
    "kaggle": "2.2.3",
    "kagglehub": "1.0.2",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, check=False, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def environment_fingerprint() -> dict:
    packages = {}
    for package in FINGERPRINT_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key] = value.strip()
    except OSError:
        memory = {"status": "unavailable"}
    fingerprint = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "kaggle_image": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_DOCKER_IMAGE"),
        "packages": packages,
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "disk": shutil.disk_usage("/")._asdict(),
    }
    try:
        import torch

        fingerprint["torch_runtime"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": (
                torch.cuda.nccl.version()
                if torch.cuda.is_available() and hasattr(torch.cuda, "nccl")
                else None
            ),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                    "vram": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # fingerprinting must describe, not hide, failure
        fingerprint["torch_runtime_error"] = type(exc).__name__
    return fingerprint


def assert_two_t4(fingerprint: dict) -> None:
    runtime = fingerprint.get("torch_runtime", {})
    names = [gpu.get("name", "") for gpu in runtime.get("gpus", [])]
    if len(names) != 2 or any("T4" not in name.upper() for name in names):
        raise RuntimeError(f"DDP run requires exactly 2 x NVIDIA T4; found {names}")


def compatibility_matrix(before: dict, after: dict) -> dict:
    """Fail closed on repo pins and confirm torch was not replaced in-kernel."""
    rows = {}
    actual = after.get("packages", {})
    for package, required in REQUIRED_KAGGLE_PACKAGES.items():
        found = actual.get(package)
        rows[package] = {
            "required": required,
            "actual": found,
            "status": "compatible" if found == required else "mismatch",
        }
    before_torch = before.get("torch_runtime", {}).get("version")
    after_torch = after.get("torch_runtime", {}).get("version")
    rows["torch_image_pair"] = {
        "required": "preserve Kaggle torch/torchvision/CUDA",
        "actual": {
            "before_torch": before_torch,
            "after_torch": after_torch,
            "torchvision": actual.get("torchvision"),
            "cuda": after.get("torch_runtime", {}).get("cuda"),
        },
        "status": "compatible" if before_torch and before_torch == after_torch else "mismatch",
    }
    mismatches = [name for name, row in rows.items() if row["status"] != "compatible"]
    if mismatches:
        raise RuntimeError(f"Kaggle dependency compatibility mismatch: {mismatches}")
    return rows


def load_kaggle_secrets(secret_names: tuple[str, ...], secret_dir: str | Path) -> None:
    """Load named Kaggle secrets into the process without printing values."""
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    values = {}
    for name in secret_names:
        value = client.get_secret(name)
        if not value:
            raise RuntimeError(f"Required Kaggle secret {name!r} is empty")
        values[name] = value

    os.environ["WANDB_API_KEY"] = values["WANDB_API_KEY"]
    os.environ["HF_TOKEN"] = values["HF_TOKEN"]
    os.environ["KAGGLE_API_TOKEN"] = values["KAGGLE_API_TOKEN"]
    try:
        service_account = json.loads(values["GCS_SERVICE_ACCOUNT"])
    except json.JSONDecodeError as exc:
        raise RuntimeError("GCS_SERVICE_ACCOUNT is not valid JSON") from exc
    required = {"type", "project_id", "private_key", "client_email"}
    if required - set(service_account):
        raise RuntimeError("GCS_SERVICE_ACCOUNT is missing required fields")

    secret_dir = Path(secret_dir)
    secret_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    key_path = secret_dir / "gcs-service-account.json"
    key_path.write_text(json.dumps(service_account), encoding="utf-8")
    key_path.chmod(0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)


def discover_dataset(dataset_slug: str, input_root: str | Path = "/kaggle/input") -> Path:
    """Resolve exactly one shallow mount by slug and required manifest."""
    input_root = Path(input_root)
    slug = dataset_slug.split("/")[-1].strip()
    candidates = []
    for child in input_root.iterdir():
        if not child.is_dir() or child.name != slug:
            continue
        if (child / "dataset_manifest.json").is_file():
            candidates.append(child)
        elif (child / "manifests" / "dataset_manifest.json").is_file():
            candidates.append(child)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one mounted dataset {slug!r} with a manifest; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def load_dataset_manifest(dataset_root: str | Path) -> tuple[dict, Path, str]:
    root = Path(dataset_root)
    candidates = [
        root / "dataset_manifest.json",
        root / "manifests" / "dataset_manifest.json",
    ]
    paths = [path for path in candidates if path.is_file()]
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one dataset_manifest.json under {root}")
    path = paths[0]
    return json.loads(path.read_text(encoding="utf-8")), path, sha256_file(path)


def write_runtime_env_config(dataset_root: str | Path, output_dir: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    payload = {
        "paths": {
            "dataset_root": str(root),
            "mimic_cxr_jpg_root": str(root),
            "chexpert_csv": str(root / "mimic-cxr-2.0.0-chexpert-subset.csv.gz"),
            "processed_train_csv": str(root / "processed" / "train.csv"),
            "processed_val_csv": str(root / "processed" / "val.csv"),
            "processed_test_csv": str(root / "processed" / "test.csv"),
            "output_dir": str(Path(output_dir).resolve()),
        },
        "wandb": {"entity": "", "project": "meta-cxr"},
    }
    path = Path("configs/env_config.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def config_fingerprint(config_path: str | Path, overrides: dict) -> str:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    payload = {"config": config, "overrides": overrides}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_session_eta(
    observed_steps: int,
    observed_seconds: float,
    total_steps: int,
    session_hours: float = 12.0,
    upload_reserve_minutes: float = 90.0,
) -> dict:
    if observed_steps <= 0 or observed_seconds <= 0:
        raise ValueError("ETA probe must contain positive steps and duration")
    seconds_per_step = observed_seconds / observed_steps
    predicted = seconds_per_step * total_steps
    budget = session_hours * 3600 - upload_reserve_minutes * 60
    result = {
        "seconds_per_optimizer_step": seconds_per_step,
        "predicted_train_seconds": predicted,
        "train_budget_seconds": budget,
        "fits": predicted <= budget,
    }
    if not result["fits"]:
        raise RuntimeError(
            "Measured epoch ETA exceeds the session budget with upload reserve; "
            "reduce the deterministic subject subset and rebuild the manifest."
        )
    return result
