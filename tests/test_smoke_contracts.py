import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_training_config_is_exact_e123_multiview_policy():
    config = yaml.safe_load(
        (ROOT / "pretraining/configs/stage1_smoke_2xt4.yaml").read_text()
    )
    assert config["model"]["encoders"] == {
        "biovil": True,
        "pubmedclip": True,
        "swin": True,
    }
    assert config["model"]["multi_view"] is True
    assert config["run"]["world_size"] == 2
    assert config["run"]["distributed"] is True
    # Checkpoint selection is on positive-class macro F1 (max), matching the
    # metric nb01 records in config_fingerprint. Total validation loss was the
    # previous selector and it is dominated by ITC, which rose across epochs
    # 0-2 of the e123 run while F1 doubled -- so `loss` pinned
    # checkpoint_best to epoch 0, the worst classifier of the three.
    assert config["run"]["selection_metric"] == "f1_positive_macro_defined_only"
    assert config["run"]["selection_mode"] == "max"
    assert config["run"]["uncertain_policy"] == "ignore_uncertain"


def test_sensitivity_matrix_has_seven_predeclared_masks():
    source = (ROOT / "scripts/evaluate_encoder_sensitivity.py").read_text()
    for run_id in ("E1", "E2", "E3", "E12", "E13", "E23", "E123"):
        assert f'"{run_id}"' in source
    assert "not causal contribution" in source


def test_resume_contract_keeps_rng_and_explicit_sampler_epoch():
    runner = (ROOT / "model/lavis/runners/runner_base.py").read_text()
    resume = (ROOT / "smoke/resume.py").read_text()
    loader = (
        ROOT / "model/lavis/datasets/datasets/dataloader_utils.py"
    ).read_text()
    # Runner persists per-rank RNG and delegates capture/restore to the shared
    # helper, and MUST load the checkpoint on CPU so the RNG ByteTensors stay
    # restorable (map_location=cuda was the source of the ByteTensor failure).
    assert '"rng_by_rank"' in runner
    assert "capture_rng_state" in runner
    assert "restore_rng_state" in runner
    assert 'map_location="cpu"' in runner
    assert "map_location=self.device" not in runner
    # The actual RNG restore lives in smoke/resume.py, coercing to CPU ByteTensor.
    assert "random.setstate" in resume
    assert "np.random.set_state" in resume
    assert "torch.set_rng_state" in resume
    assert "torch.cuda.set_rng_state_all" in resume
    # Data-order continuity: the sampler epoch is set explicitly per runner epoch.
    assert "def set_epoch" in loader
    assert "sampler.set_epoch" in loader


def test_notebooks_are_clean_and_parameterized():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "01_stage1_smoke_2xt4.ipynb",
        "02_encoder_sensitivity_2xt4.ipynb",
        "03_report_generation_metrics_2xt4.ipynb",
    ]
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
        source = "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])
        assert "REPO_COMMIT = " in source
        assert "GCS_SERVICE_ACCOUNT" in source
        assert "WANDB_API_KEY" in source
        assert "HF_TOKEN" not in source
        assert "KAGGLE_API_TOKEN" not in source
