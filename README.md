# AdvSoli — adversarial training for Soli radar gesture recognition

ELL784 / ELL7286 (Intro to Machine Learning), IIT Delhi · Assignment 3.

This repo holds my code for Assignment 3. The dataset is Google's Soli
range-Doppler radar (32×32 RDM frames, 4 channels, 11 gestures, 10
subjects) and the task is two-fold:

1. lift performance on the four fine-grained classes (finger slider,
   finger rub, pinch index, pinch pinky), and
2. generalize across subjects (two-fold subject CV).

I built a small custom encoder (no pretrained weights) called **AdvSoli**
that bolts two adversarial signals on top of a 2D-CNN + Bi-GRU backbone:
- a **conditional GAN** (cGAN) that synthesizes extra range-Doppler
  clips for the four hard classes — used as augmentation for the
  classifier, and
- a **PGD adversarial attack** on the input during training so the
  encoder learns input-robust features.

I also ran a third adversarial signal — a **DANN subject discriminator
with a Gradient Reversal Layer** — but it consistently hurt accuracy in
this small-data setting, so I dropped it from the final model and report
it as a negative result. See `report/main.tex` for the full write-up.

Headline results on two-fold cross-subject CV (mean over folds):

| Variant                  | overall acc | fine-grained acc | PGD-robust |
|--------------------------|-------------|------------------|------------|
| Backbone (no adv.)       | 86.5        | 80.9             | 2.4        |
| + cGAN only              | 86.6        | **82.2**         | 0.1        |
| + PGD only               | **88.4**    | 81.4             | **27.3**   |
| **AdvSoli (cGAN + PGD)** | 86.6        | 80.3             | **21.9**   |

AdvSoli matches the strong backbone clean accuracy while making the
encoder ~9× more robust to a 10-step PGD attack with `eps=0.03`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested on macOS (Apple Silicon, MPS backend) and Linux (CPU/CUDA).

## Get the data

The Soli range-Doppler files are mirrored by the deep-soli authors:

```bash
bash scripts/download.sh
```

This pulls and unzips the archive into `data/dsp/`, which will end up
with 5,500 `.h5` files. Each file has datasets `ch0..ch3` (one per radar
receiver) and a `label` array.

Then build the canonical 10-subject 2-fold split:

```bash
python scripts/make_splits.py --root data/dsp --out configs/splits.json
```

By default this restricts to the canonical cross-user sessions
{2,3,5,6,8,9,10,11,12,13}, which gives exactly 2,750 clips and 10
subjects.

## Train

```bash
# fold 0 (train subjects {2,3,5,6,8}, test {9,10,11,12,13})
python -m src.train --config configs/default.yaml --fold 0

# fold 1 (swap)
python -m src.train --config configs/default.yaml --fold 1
```

Quick sanity run (3 epochs):

```bash
python -m src.train --config configs/default.yaml --fold 0 --quick
```

If you don't have the data downloaded and just want to test the pipeline:

```bash
python -m src.train --quick --synthetic --fold 0
```

## Reproduce the ablation

The ablation table in the report comes from:

```bash
# 5 variants × 2 folds = 10 runs, ≈ 2.5 hours on Apple Silicon
python scripts/run_ablations.py

# the proposed AdvSoli (cGAN+PGD) variant on both folds, ≈ 30 min
python scripts/run_advsoli.py
```

Both write a summary file at `runs/ablation_summary.json` and per-run
training logs at `runs/<variant>_fold<k>_log.json`.

The figures and tables are then made by:

```bash
python scripts/plot_results.py     # curves + per-class bar + ablation tsv
python scripts/plot_confusion.py   # confusion matrix figure
```

## Evaluate a checkpoint

```bash
# clean + PGD-robust accuracy + per-class table
python -m src.eval --ckpt checkpoints/advsoli_fold0_best.pt
```

Set `--pgd_eps 0` to skip the adversarial robustness eval.

## Trained checkpoints

After running `scripts/run_ablations.py` and `scripts/run_advsoli.py`,
the released checkpoints live in `checkpoints/`:

| File                          | Variant            | Fold |
|-------------------------------|--------------------|------|
| `advsoli_fold0_best.pt`       | AdvSoli (cGAN+PGD) | 0    |
| `advsoli_fold1_best.pt`       | AdvSoli (cGAN+PGD) | 1    |
| `backbone_fold{0,1}_best.pt`  | Backbone           | 0/1  |
| `pPGD_fold{0,1}_best.pt`      | + PGD only         | 0/1  |
| `pcGAN_fold{0,1}_best.pt`     | + cGAN only        | 0/1  |
| `pDANN_fold{0,1}_best.pt`     | + DANN only        | 0/1  |
| `full_fold{0,1}_best.pt`      | cGAN+PGD+DANN      | 0/1  |

Each checkpoint stores the encoder, classifier, subject head, generator
and discriminator state dicts plus the config used to train.

**External model download (Google Drive):**
[All checkpoints (advsoli, backbone, ablation variants)](https://drive.google.com/drive/folders/1GEa77k0CFr8-449y5f3Idb23L8PGKLDO?usp=sharing)

## Repo layout

```
src/
  data_loader.py    Soli HDF5 dataset + 2-fold split loader + synthetic stub
  model.py          encoder, classifier, subject head, cGAN G + D
  losses.py         CE, BCE-with-logits, GAN G/D, DANN lambda schedule (from scratch)
  adversarial.py    Gradient Reversal Layer + PGD attack (from scratch)
  train.py          training loop with cls + DANN + cGAN + PGD losses
  eval.py           standalone evaluator (clean + per-class + PGD-robust)
  utils.py          device picker, seeding, meter, yaml loader
configs/
  default.yaml      all hyperparameters in one place; flip loss weights to ablate
  splits.json       per-fold subject split (built by make_splits.py)
scripts/
  download.sh       fetches the Soli HDF5 archive
  make_splits.py    builds configs/splits.json
  run_ablations.py  5 ablation variants × 2 folds
  run_advsoli.py    proposed cGAN+PGD model × 2 folds
  plot_results.py   training curves + ablation table + per-class bar
  plot_confusion.py confusion matrix figure
report/
  main.tex          CVPR-format write-up
  refs.bib          references
  figs/             plots from scripts/plot_*.py
```

## Notes on implementation

- All loss functions, the GRL, the PGD attack and the GAN training
  loop are written from scratch in `src/losses.py` and
  `src/adversarial.py`. We do not use `nn.BCEWithLogitsLoss` or any
  pre-built combined loss.
- No pretrained networks are used anywhere.
- The encoder + classifier + subject head + generator + discriminator
  total 1.6M parameters.
- Default config trains in about 10–15 min per fold on Apple Silicon
  (MPS); a full ablation sweep takes ~3 hours.

## License / academic note

This is coursework — please don't copy it into your own assignment
submission.
