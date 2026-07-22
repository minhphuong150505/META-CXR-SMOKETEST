# META-CXR Stage-1 smoke test

Minimal Stage-1 closure sourced from
[`minhphuong150505/Meta-CXR`](https://github.com/minhphuong150505/Meta-CXR) at
commit `0cc1ccd69fb1f7b1d5084aaea788fad274725600`.

MIMIC-CXR is credentialed-access data. Never make the dataset, reports,
per-sample predictions, feature caches, or checkpoints public. This repository
contains code only.

Attach exactly one private Kaggle dataset containing `dataset_manifest.json`
and the documented `files/`, `processed/`, and `manifests/` layout. Set the
parameter cell in `notebooks/01_stage1_smoke_2xt4.ipynb`, run one session epoch,
then publish the verified output to the private checkpoint dataset handle.
Resume by attaching that checkpoint dataset and incrementing `SESSION_INDEX`.
Configure the private Kaggle secrets `GCS_SERVICE_ACCOUNT`, `WANDB_API_KEY`,
`HF_TOKEN`, and `KAGGLE_API_TOKEN`; the last one authorizes private checkpoint
dataset updates.

`notebooks/02_encoder_sensitivity_2xt4.ipynb` evaluates seven post-training
encoder masks from the single E123 checkpoint. These are inference-sensitivity
measurements, not independently trained ablations and not causal contributions.

Private run artifacts use:

```text
<run>/checkpoint_best.pth
<run>/checkpoint_last.pth
<run>/result/*.json
<run>/run_manifest.json
```
