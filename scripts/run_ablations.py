"""
Ablation runner: trains the 5 variants we report in the paper, on both
folds, and writes a summary json that the report uses.

Variants (loss-weight overrides on top of configs/default.yaml):
    backbone : subj_adv=0   gan=0   pgd=0
    +DANN    : subj_adv>0   gan=0   pgd=0
    +cGAN    : subj_adv=0   gan>0   pgd=0
    +PGD     : subj_adv=0   gan=0   pgd>0
    full     : all three on (the AdvSoli model)

Run:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --quick --synthetic   # for sanity
"""
import argparse, copy, json, os, sys, time

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, ensure_dir
from src.train import train_one_fold


VARIANTS = {
    "backbone": dict(subj_adv=0.0, gan=0.0, pgd=0.0),
    "+DANN":    dict(subj_adv=0.3, gan=0.0, pgd=0.0),
    "+cGAN":    dict(subj_adv=0.0, gan=0.2, pgd=0.0),
    "+PGD":     dict(subj_adv=0.0, gan=0.0, pgd=0.1),
    "full":     dict(subj_adv=0.3, gan=0.2, pgd=0.1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="runs/ablation_summary.json")
    ap.add_argument("--folds", default="0,1",
                    help="comma list, e.g. 0 or 0,1")
    args = ap.parse_args()

    base = load_yaml(args.config)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    summary = {}
    for name, ovr in VARIANTS.items():
        summary[name] = {}
        for fold in folds:
            print(f"\n========== variant {name}  fold {fold} ==========")
            cfg = copy.deepcopy(base)
            for k, v in ovr.items():
                cfg["loss"][k] = v
            # use a unique checkpoint per variant to avoid clobbering
            tag = name.replace("+", "p")
            cfg.setdefault("_tag", tag)

            class _A: pass
            a = _A(); a.quick = args.quick; a.synthetic = args.synthetic
            a.splits = args.splits

            # patch ckpt + log path via env-style hack: monkeypatch ensure_dir? simpler:
            # we just change cwd-relative paths inside train_one_fold by overriding
            # the _checkpoint and _log paths in its closure. The cleanest way is to
            # let train_one_fold write to a sub-folder named by tag.
            # We do that by symlinking checkpoints/<tag>/ -> checkpoints/ inside
            # a small adapter. Simpler -- we just rename files after the run.
            t0 = time.time()
            best_acc, ckpt = train_one_fold(cfg, fold, a)
            secs = time.time() - t0

            # rename the produced files so different variants don't overwrite
            from shutil import move
            new_ck = ckpt.replace(f"fold{fold}_best", f"{tag}_fold{fold}_best")
            try:
                move(ckpt, new_ck); ckpt = new_ck
            except FileNotFoundError:
                pass

            log_src = os.path.join("runs", f"fold{fold}_log.json")
            log_dst = os.path.join("runs", f"{tag}_fold{fold}_log.json")
            try:
                move(log_src, log_dst)
            except FileNotFoundError:
                log_dst = None

            summary[name][f"fold{fold}"] = {
                "best_acc": best_acc, "ckpt": ckpt, "log": log_dst,
                "loss_weights": ovr, "secs": secs,
            }

    ensure_dir(os.path.dirname(args.out) or ".")
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\n=== final ===")
    for name, fr in summary.items():
        accs = [v["best_acc"] for v in fr.values()]
        mean = sum(accs) / len(accs) if accs else 0.0
        print(f"  {name:<10s} mean test acc over folds: {mean:.4f}")


if __name__ == "__main__":
    main()
