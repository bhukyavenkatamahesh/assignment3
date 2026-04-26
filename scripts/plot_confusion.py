"""
Confusion matrix figure for the report.
Compares backbone vs AdvSoli on fold 0.
"""
import os, json, torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import pick_device
from src.data_loader import SoliDataset, load_split
from src.model import build_advsoli
from src.eval import GESTURE_NAMES, run_eval


def cm_for(ckpt, fold):
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = st['config']
    dev = pick_device(cfg['train']['device'])
    enc, cls, _, _, _ = build_advsoli(cfg, num_classes=11, num_subjects=10)
    enc.load_state_dict(st['encoder']); cls.load_state_dict(st['classifier'])
    enc, cls = enc.to(dev), cls.to(dev)
    _, te, sess2subj = load_split('configs/splits.json', fold)
    ds = SoliDataset(cfg['data']['root'], te, seq_len=cfg['data']['seq_len'],
                     channels=tuple(cfg['data']['channels']), sess2subj=sess2subj,
                     normalize=True, in_memory=True)
    dl = DataLoader(ds, batch_size=16)
    m = run_eval(enc, cls, dl, dev,
                 fine_classes=cfg['adversarial']['fine_grained_classes'],
                 pgd_eps=0.0)
    return np.asarray(m['confusion'], dtype=np.float64)


def normalize_rows(cm):
    rs = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, rs, out=np.zeros_like(cm), where=rs>0)


def plot_one(ax, cm, title):
    nc = normalize_rows(cm)
    im = ax.imshow(nc, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(GESTURE_NAMES)))
    ax.set_yticks(range(len(GESTURE_NAMES)))
    ax.set_xticklabels(GESTURE_NAMES, rotation=40, ha='right', fontsize=7)
    ax.set_yticklabels(GESTURE_NAMES, fontsize=7)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    for i in range(nc.shape[0]):
        for j in range(nc.shape[1]):
            v = nc[i,j]
            if v > 0.05:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, color='white' if v > 0.5 else 'black')
    return im


def main():
    os.makedirs('report/figs', exist_ok=True)
    cm_b = cm_for('checkpoints/backbone_fold0_best.pt', 0)
    cm_a = cm_for('checkpoints/advsoli_fold0_best.pt', 0)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_one(axs[0], cm_b, 'backbone (fold 0)')
    im = plot_one(axs[1], cm_a, 'AdvSoli (cGAN+PGD), fold 0')
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    fig.savefig('report/figs/confusion.pdf', bbox_inches='tight')
    fig.savefig('report/figs/confusion.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote report/figs/confusion.pdf')


if __name__ == '__main__':
    main()
