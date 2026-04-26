# Assignment-3: AdvSoli — adversarial training for Soli radar gesture recognition

ELL784 / ELL7286 (Intro to Machine Learning), IIT Delhi.

This repo holds my implementation for Assignment-3. The goal is two-fold:
1. Boost recognition of *fine-grained* gestures on the Soli range-Doppler radar
   data (finger slider, finger rub, pinch index, pinch pinky).
2. Improve *cross-subject* generalization using two-fold subject CV.

I take a single, unified model which I call **AdvSoli** that combines three
adversarial signals on top of a custom temporal-CNN encoder:
- a **subject discriminator with a Gradient Reversal Layer** (DANN-style)
  to push the encoder towards subject-invariant features,
- a small **conditional GAN** that synthesizes extra range-Doppler clips
  for the four hard classes,
- and **PGD adversarial perturbations** added to the radar input during
  training for robustness.

(Full method description and ablations are in `report/main.tex`.)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested on macOS (Apple Silicon, MPS backend) and Linux (CPU/CUDA).

## Get the data

Soli is hosted by the deep-soli authors on polybox.ethz.ch:

```bash
bash scripts/download.sh
```

After unzipping, `data/dsp/` will contain files named
`<gesture>_<session>_<instance>.h5` with four channels of 32x32 RDM frames.

Then build the 2-fold subject split:
```bash
python scripts/make_splits.py --root data/dsp --out configs/splits.json
```

## Train

```bash
# fold 0 (train subjects 0..4, test 5..9)
python -m src.train --config configs/default.yaml --fold 0

# fold 1 (swap)
python -m src.train --config configs/default.yaml --fold 1
```

Quick sanity run:
```bash
python -m src.train --config configs/default.yaml --fold 0 --quick
```

## Evaluate

```bash
python -m src.eval --config configs/default.yaml --ckpt checkpoints/fold0_best.pt
```

## Pretrained model

Final model (fold-0): _link to be added after training_.

## Repo layout

```
src/             core code (data loader, model, losses, train, eval)
configs/         yaml configs and the subject split
scripts/         data download + split builder
report/          CVPR-format writeup
```
