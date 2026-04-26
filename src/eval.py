"""
Standalone evaluator. Loads a checkpoint and reports:
  - overall accuracy
  - fine-grained class accuracy (mean over [pinch_idx, pinch_pinky,
                                             finger_slider, finger_rub])
  - per-class accuracy
  - confusion matrix
  - PGD-attack accuracy (a small robustness check)

Usage:
    python -m src.eval --ckpt checkpoints/fold0_best.pt
    python -m src.eval --ckpt checkpoints/fold0_best.pt --synthetic
"""
import argparse, json, os
import numpy as np
import torch

from .utils import pick_device
from .data_loader import SoliDataset, SyntheticSoli, load_split
from .model import build_advsoli
from .adversarial import pgd_attack
from torch.utils.data import DataLoader


GESTURE_NAMES = [
    "pinch_index", "pinch_pinky", "finger_slider", "finger_rub",
    "slow_swipe", "fast_swipe", "push", "pull",
    "palm_tilt", "circle", "palm_hold",
]


def load_test_loader(cfg, fold, splits, synthetic):
    if synthetic:
        ds = SyntheticSoli(n=80, seq_len=cfg['data']['seq_len'],
                           num_classes=cfg['data']['num_classes'],
                           num_subjects=cfg['data']['num_subjects'], seed=999)
    else:
        _, te, sess2subj = load_split(splits, fold)
        ds = SoliDataset(cfg['data']['root'], te,
                         seq_len=cfg['data']['seq_len'],
                         channels=tuple(cfg['data']['channels']),
                         sess2subj=sess2subj, normalize=True)
    return DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=False)


def confusion(preds, labels, K):
    cm = np.zeros((K, K), dtype=np.int64)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm


def run_eval(enc, cls, loader, device, fine_classes,
             pgd_eps=0.0, pgd_alpha=0.01, pgd_iters=3):
    enc.eval(); cls.eval()
    all_pred, all_lab = [], []
    adv_correct = adv_total = 0

    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        with torch.no_grad():
            pred = cls(enc(x)).argmax(1)
        all_pred.append(pred.cpu().numpy())
        all_lab.append(y.cpu().numpy())

        if pgd_eps > 0:
            fwd = lambda xx: cls(enc(xx))
            x_adv = pgd_attack(fwd, x, y, eps=pgd_eps,
                               alpha=pgd_alpha, iters=pgd_iters)
            with torch.no_grad():
                pa = cls(enc(x_adv)).argmax(1)
            adv_correct += int((pa == y).sum().item())
            adv_total += int(y.numel())

    preds = np.concatenate(all_pred)
    labs  = np.concatenate(all_lab)
    K = len(GESTURE_NAMES)
    cm = confusion(preds, labs, K)

    overall = float((preds == labs).mean())
    per_class = []
    for k in range(K):
        m = labs == k
        per_class.append(float((preds[m] == labs[m]).mean()) if m.any() else 0.0)
    fg_mask = np.isin(labs, list(fine_classes))
    fg = float((preds[fg_mask] == labs[fg_mask]).mean()) if fg_mask.any() else 0.0
    adv_acc = adv_correct / max(1, adv_total) if pgd_eps > 0 else None
    return {
        "overall_acc": overall,
        "fine_grained_acc": fg,
        "per_class_acc": per_class,
        "confusion": cm.tolist(),
        "pgd_eps": pgd_eps,
        "pgd_acc": adv_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--pgd_eps", type=float, default=0.03,
                    help="set 0 to skip the adversarial robustness eval")
    ap.add_argument("--out", default=None,
                    help="optional path to dump the metrics json")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("config")
    if cfg is None:
        from .utils import load_yaml
        cfg = load_yaml(args.config)
    fold = args.fold if args.fold is not None else state.get("fold", 0)

    device = pick_device(cfg['train']['device'])
    enc, cls, _, _, _ = build_advsoli(
        cfg, num_classes=cfg['data']['num_classes'],
        num_subjects=cfg['data']['num_subjects'])
    enc.load_state_dict(state["encoder"])
    cls.load_state_dict(state["classifier"])
    enc, cls = enc.to(device), cls.to(device)

    loader = load_test_loader(cfg, fold, args.splits, args.synthetic)
    metrics = run_eval(
        enc, cls, loader, device,
        fine_classes=cfg['adversarial']['fine_grained_classes'],
        pgd_eps=args.pgd_eps,
        pgd_alpha=cfg['adversarial']['pgd_alpha'],
        pgd_iters=cfg['adversarial']['pgd_iters'],
    )

    print(f"\nfold {fold}  ckpt {args.ckpt}")
    print(f"  overall acc       : {metrics['overall_acc']:.4f}")
    print(f"  fine-grained acc  : {metrics['fine_grained_acc']:.4f}")
    if metrics["pgd_acc"] is not None:
        print(f"  PGD acc (eps={metrics['pgd_eps']}) : {metrics['pgd_acc']:.4f}")
    print("  per-class:")
    for k, name in enumerate(GESTURE_NAMES):
        marker = " *" if k in cfg['adversarial']['fine_grained_classes'] else "  "
        print(f"    {marker} {k:2d} {name:<14s} {metrics['per_class_acc'][k]:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
