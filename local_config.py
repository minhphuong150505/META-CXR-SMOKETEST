from pathlib import Path
from omegaconf import OmegaConf

_CONFIG_PATH = Path(__file__).parent / "configs" / "env_config.yaml"

if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"env_config.yaml not found at {_CONFIG_PATH}. "
        "Copy configs/env_config.yaml.example to configs/env_config.yaml and fill in your paths."
    )

_cfg = OmegaConf.to_container(OmegaConf.load(str(_CONFIG_PATH)), resolve=True)


def _resolve_gz_variant(path_str: str) -> str:
    """Tolerate Kaggle stripping the .gz suffix at dataset-ingest time.

    The training subprocess imports this fresh from disk, so resolving here
    fixes a stale ``.csv.gz`` path even when env_config.yaml was written by a
    cached notebook-kernel module. Falls back to the original path so the
    reader raises a clear error if neither variant exists.
    """
    path = Path(path_str)
    if path.is_file():
        return path_str
    alt = path.with_suffix("") if path.suffix == ".gz" else Path(path_str + ".gz")
    return str(alt) if alt.is_file() else path_str


VIS_ROOT          = _cfg["paths"]["mimic_cxr_jpg_root"]
CHEXPERT_CSV      = _resolve_gz_variant(_cfg["paths"]["chexpert_csv"])
PROCESSED_TRAIN_CSV = _cfg["paths"]["processed_train_csv"]
PROCESSED_VAL_CSV   = _cfg["paths"]["processed_val_csv"]
PROCESSED_TEST_CSV  = _cfg["paths"]["processed_test_csv"]
OUTPUT_DIR        = _cfg["paths"]["output_dir"]

WANDB_ENTITY      = _cfg["wandb"]["entity"]
WANDB_PROJECT     = _cfg["wandb"]["project"]
