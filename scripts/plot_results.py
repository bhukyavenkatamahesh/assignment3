"""
Reads runs/ablation_summary.json + the per-variant log files and produces
the figures + tables we use in the report.

Outputs go to report/figs/ and report/tables/.
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt


VARIANT_ORDER = ["backbone", "+DANN", "+cGAN", "+PGD", "full", "advsoli"]
GESTURES = ["pinch_idx", "pinch_pky", "fin_slid", "fin_rub",
            "slow_sw", "fast_sw", "push", "pull",
            "palm_tilt", "circle", "palm_hold"]
FINE_IDX = [0, 1, 2, 3]


def load_summary(p):
    with open(p) as f:
        return json.load(f)


def load_log(p):
    if p is None or not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="runs/ablation_summary.json")
    ap.add_argument("--out_figs", default="report/figs")
    ap.add_argument("--out_tabs", default="report/tables")
    args = ap.parse_args()

    os.makedirs(args.out_figs, exist_ok=True)
    os.makedirs(args.out_tabs, exist_ok=True)

    summ = load_summary(args.summary)

    # ---- table 1: ablation, mean over folds ----
    tbl_lines = ["variant\toverall_acc\tfine_acc"]
    for v in VARIANT_ORDER:
        if v not in summ:
            continue
        fold_results = summ[v]
        accs, fgs = [], []
        for fname, info in fold_results.items():
            log = load_log(info.get("log"))
            if not log:
                accs.append(info.get("best_acc", 0.0)); fgs.append(0.0); continue
            best = max(log, key=lambda r: r["test_acc"])
            accs.append(best["test_acc"])
            fgs.append(best["fg_acc"])
        a = np.mean(accs); af = np.mean(fgs)
        s = np.std(accs); sf = np.std(fgs)
        tbl_lines.append(f"{v}\t{a:.3f}±{s:.3f}\t{af:.3f}±{sf:.3f}")

    with open(os.path.join(args.out_tabs, "ablation.tsv"), "w") as f:
        f.write("\n".join(tbl_lines))
    print("wrote", os.path.join(args.out_tabs, "ablation.tsv"))

    # ---- figure 1: training curves on fold 0 ----
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for v in VARIANT_ORDER:
        if v not in summ:
            continue
        info = summ[v].get("fold0")
        if info is None:
            continue
        log = load_log(info.get("log"))
        if not log:
            continue
        eps = [r["epoch"] for r in log]
        accs = [r["test_acc"] for r in log]
        ax.plot(eps, accs, label=v, linewidth=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("test acc (fold 0)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_figs, "curves_fold0.pdf"))
    fig.savefig(os.path.join(args.out_figs, "curves_fold0.png"), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(args.out_figs, "curves_fold0.pdf"))

    # ---- figure 2: per-class accuracy bars (full vs backbone) for fold 0 ----
    def per_class_for(variant, fold="fold0"):
        info = summ.get(variant, {}).get(fold)
        if info is None: return None
        log = load_log(info.get("log"))
        if not log: return None
        best = max(log, key=lambda r: r["test_acc"])
        return best.get("per_class")

    # prefer advsoli (cGAN+PGD) as the headline unified model;
    # fall back to "full" if it's not run yet
    pc_back = per_class_for("backbone")
    headline = "advsoli" if per_class_for("advsoli") is not None else "full"
    pc_full = per_class_for(headline)
    if pc_back is not None and pc_full is not None:
        x = np.arange(len(GESTURES))
        w = 0.4
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        label_full = "AdvSoli (cGAN+PGD)" if headline == "advsoli" else "full (cGAN+DANN+PGD)"
        ax.bar(x - w/2, pc_back, w, label="backbone", color="#888")
        ax.bar(x + w/2, pc_full, w, label=label_full, color="#2c7fb8")
        for fi in FINE_IDX:
            ax.axvspan(fi - 0.5, fi + 0.5, color="#fdae6b", alpha=0.18, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(GESTURES, rotation=35, ha="right",
                                              fontsize=8)
        ax.set_ylabel("test acc (fold 0)")
        ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_figs, "perclass_bar.pdf"))
        fig.savefig(os.path.join(args.out_figs, "perclass_bar.png"), dpi=150)
        plt.close(fig)
        print("wrote", os.path.join(args.out_figs, "perclass_bar.pdf"))


if __name__ == "__main__":
    main()
