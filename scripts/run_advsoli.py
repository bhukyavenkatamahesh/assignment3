"""
Trains the proposed AdvSoli unified model:  cGAN + PGD (DANN dropped).
Two folds.  Outputs go alongside the existing ablation files.

The original 3-component "full" run is kept as an extended ablation to
show why we drop DANN.

Run:
    python scripts/run_advsoli.py
"""
import argparse, copy, json, os, sys, time
from shutil import move

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, ensure_dir
from src.train import train_one_fold


TAG = "advsoli"   # = +cGAN +PGD, no DANN
OVERRIDE = dict(subj_adv=0.0, gan=0.2, pgd=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--summary", default="runs/ablation_summary.json")
    args = ap.parse_args()

    base = load_yaml(args.config)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    # load existing summary so we can append
    summary = {}
    if os.path.exists(args.summary):
        with open(args.summary) as f:
            summary = json.load(f)
    summary.setdefault(TAG, {})

    class _A: pass
    a = _A(); a.quick = False; a.synthetic = False; a.splits = args.splits

    for fold in folds:
        print(f"\n========== {TAG}  fold {fold} ==========")
        cfg = copy.deepcopy(base)
        for k, v in OVERRIDE.items():
            cfg["loss"][k] = v

        t0 = time.time()
        best_acc, ckpt = train_one_fold(cfg, fold, a)
        secs = time.time() - t0

        new_ck = ckpt.replace(f"fold{fold}_best", f"{TAG}_fold{fold}_best")
        try:
            move(ckpt, new_ck); ckpt = new_ck
        except FileNotFoundError:
            pass
        log_src = os.path.join("runs", f"fold{fold}_log.json")
        log_dst = os.path.join("runs", f"{TAG}_fold{fold}_log.json")
        try:
            move(log_src, log_dst)
        except FileNotFoundError:
            log_dst = None

        summary[TAG][f"fold{fold}"] = {
            "best_acc": best_acc, "ckpt": ckpt, "log": log_dst,
            "loss_weights": OVERRIDE, "secs": secs,
        }

    ensure_dir(os.path.dirname(args.summary) or ".")
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nappended to {args.summary}")


if __name__ == "__main__":
    main()
