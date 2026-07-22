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
    assert config["run"]["selection_metric"] == "f1_positive_macro_defined_only"
    assert config["run"]["uncertain_policy"] == "ignore_uncertain"


def test_sensitivity_matrix_has_seven_predeclared_masks():
    source = (ROOT / "scripts/evaluate_encoder_sensitivity.py").read_text()
    for run_id in ("E1", "E2", "E3", "E12", "E13", "E23", "E123"):
        assert f'"{run_id}"' in source
    assert "not causal contribution" in source


def test_resume_contract_keeps_rng_and_explicit_sampler_epoch():
    runner = (ROOT / "model/lavis/runners/runner_base.py").read_text()
    loader = (
        ROOT / "model/lavis/datasets/datasets/dataloader_utils.py"
    ).read_text()
    assert '"rng_by_rank"' in runner
    assert "random.setstate" in runner
    assert "np.random.set_state" in runner
    assert "torch.set_rng_state" in runner
    assert "def set_epoch" in loader
    assert "sampler.set_epoch" in loader


def test_notebooks_are_clean_and_parameterized():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "01_stage1_smoke_2xt4.ipynb",
        "02_encoder_sensitivity_2xt4.ipynb",
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
        assert "HF_TOKEN" in source
        assert "KAGGLE_API_TOKEN" in source
