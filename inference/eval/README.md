# Eval Pipeline (vast.ai)

Full test-set SI-SNR / SI-SNRi evaluation for paper-faith models.

## Pre-req

```bash
bash scripts/setup_datasets.sh   # raw + 2spk + 3spk synthetic + deps
```

`gdown` and `boto3` already in setup_datasets.sh deps.

## Files

| File | Role |
|------|------|
| `best-model-list.txt` | `model_name=gdrive_file_id` per line |
| `run_eval.py`         | download → load → eval → CSV + audio + stats |
| `zip_and_upload.sh`   | zip per model → R2 upload |

## Run

```bash
# all models
python inference/eval/run_eval.py

# subset
python inference/eval/run_eval.py --models 2speaker-skim 3speaker-skim

# override audio save count (default 450)
python inference/eval/run_eval.py --audio-limit 100
AUDIO_LIMIT=100 python inference/eval/run_eval.py
```

## Output layout

```
inference/eval/
├── ckpts/<model>/best_model.pth         # cached download
└── results/
    ├── summary.json                     # all models combined
    └── <model>/
        ├── per_file.csv                 # 3600 rows: file_id,sisnr,sisnri,mix_sisnr,perm
        ├── stats.json                   # mean/std/median for this model
        └── audio/<file_id>/
            ├── mixture.wav
            ├── s1_gt.wav, s2_gt.wav (+ s3_gt.wav)
            └── s1_est.wav, s2_est.wav (+ s3_est.wav)   # reordered to PIT-best perm
```

Audio saved for first 450 mixtures per model (sorted).

## Zip + upload

```bash
# all models present in results/
bash inference/eval/zip_and_upload.sh

# subset
bash inference/eval/zip_and_upload.sh 2speaker-skim 3speaker-skim

# zip only, skip upload
NO_UPLOAD=1 bash inference/eval/zip_and_upload.sh
```

R2 destination: `$R2_PREFIX/eval/<model>.zip`.

## Name parsing rules

| Pattern | Inferred |
|---|---|
| prefix `2speaker-` | num_spk=2 |
| prefix `3speaker-` | num_spk=3 |
| contains `skim-attention` | arch=v3 (SkiMAttentionV3Separator) |
| else | arch=skim (SkiMSeparator) |

Config hardcoded to paper-faith: K=150, dropout=0.1, layer=4, unit=256, mem_type=hc, seg_overlap=False, num_heads=4 (v3 only).

## Notes

- Resume-safe: per-model `summary.json` incrementally written; if a model crashes, others continue.
- Permutation column in CSV records PIT-aligned assignment for reproducibility.
- Estimated WAVs are saved already **reordered to GT order** (so `s1_est.wav` corresponds to `s1_gt.wav`).
- Re-running won't re-download checkpoints (cached under `ckpts/`). Delete the cache dir to force fresh download.
