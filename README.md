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
then publish the verified checkpoints to the private GCS bucket configured in
the notebook. Each later session downloads `checkpoint_last.pth` from that
bucket and increments `SESSION_INDEX`.
Configure the private Kaggle secrets `GCS_SERVICE_ACCOUNT` and `WANDB_API_KEY`.
The service account must be able to read and update the private GCS checkpoint
bucket used by the notebook.

`notebooks/02_encoder_sensitivity_2xt4.ipynb` evaluates the six Table-5
post-training
encoder subsets from the single E123 checkpoint. For this evaluation only,
inactive token spans are removed before MHCAC. These are inference-sensitivity
measurements, not independently trained ablations and not causal contributions.
The notebook keeps its intermediate JSON under `/tmp` and displays the table;
it does not publish a Kaggle results dataset.

Private run artifacts use:

```text
<run>/checkpoint_best.pth
<run>/checkpoint_last.pth
<run>/result/*.json
<run>/run_manifest.json
```
