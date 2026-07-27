import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from smoke.runtime import (
    assert_session_eta,
    discover_dataset,
    load_dataset_manifest,
    load_kaggle_secrets,
    sha256_file,
    write_runtime_env_config,
)


def test_kaggle_secrets_refuse_published_working_directory():
    with pytest.raises(ValueError, match="must not be written under /kaggle/working"):
        load_kaggle_secrets(("GCS_SERVICE_ACCOUNT",), "/kaggle/working/secrets")


def test_kaggle_secret_loader_only_requires_requested_names(tmp_path, monkeypatch):
    values = {
        "GCS_SERVICE_ACCOUNT": json.dumps(
            {
                "type": "service_account",
                "project_id": "test-project",
                "private_key": "synthetic-test-key",
                "client_email": "test@example.invalid",
            }
        ),
        "WANDB_API_KEY": "synthetic-wandb-key",
    }

    class FakeSecretsClient:
        def get_secret(self, name):
            return values.get(name)

    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        SimpleNamespace(UserSecretsClient=FakeSecretsClient),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    load_kaggle_secrets(
        ("GCS_SERVICE_ACCOUNT", "WANDB_API_KEY"), tmp_path / "private"
    )

    assert "HF_TOKEN" not in os.environ
    assert "KAGGLE_API_TOKEN" not in os.environ
    assert os.environ["WANDB_API_KEY"] == "synthetic-wandb-key"
    assert Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).is_file()


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


@pytest.mark.parametrize(
    "mounted_name",
    ["mimic-cxr-2.0.0-chexpert-subset.csv.gz", "mimic-cxr-2.0.0-chexpert-subset.csv"],
)
def test_env_config_resolves_chexpert_csv_with_or_without_gz(
    tmp_path, monkeypatch, mounted_name
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / mounted_name).write_bytes(b"stub")
    monkeypatch.chdir(tmp_path)
    path = write_runtime_env_config(dataset, tmp_path / "out")
    paths = yaml.safe_load(path.read_text(encoding="utf-8"))["paths"]
    assert paths["chexpert_csv"] == str(dataset.resolve() / mounted_name)


def test_env_config_fails_fast_when_chexpert_csv_absent(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="chexpert-subset"):
        write_runtime_env_config(dataset, tmp_path / "out")


def test_eta_guard_reserves_upload_window():
    result = assert_session_eta(2, 20.0, 100, 12.0, 90.0)
    assert result["fits"] is True
    with pytest.raises(RuntimeError, match="exceeds"):
        assert_session_eta(1, 1000.0, 100, 12.0, 90.0)
