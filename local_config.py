from pathlib import Path
from omegaconf import OmegaConf

_CONFIG_PATH = Path(__file__).parent / "configs" / "env_config.yaml"

if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"env_config.yaml not found at {_CONFIG_PATH}. "
        "Copy configs/env_config.yaml.example to configs/env_config.yaml and fill in your paths."
    )

_cfg = OmegaConf.to_container(OmegaConf.load(str(_CONFIG_PATH)), resolve=True)

VIS_ROOT          = _cfg["paths"]["mimic_cxr_jpg_root"]
CHEXPERT_CSV      = _cfg["paths"]["chexpert_csv"]
PROCESSED_TRAIN_CSV = _cfg["paths"]["processed_train_csv"]
PROCESSED_VAL_CSV   = _cfg["paths"]["processed_val_csv"]
PROCESSED_TEST_CSV  = _cfg["paths"]["processed_test_csv"]
OUTPUT_DIR        = _cfg["paths"]["output_dir"]

WANDB_ENTITY      = _cfg["wandb"]["entity"]
WANDB_PROJECT     = _cfg["wandb"]["project"]
