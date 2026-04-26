"""
AdvSoli training loop.

One forward step roughly does:
  1) cls loss on real clips
  2) optional PGD adversarial step  (push features to be input-robust)
  3) optional DANN subject head     (push features to be subject-invariant)
  4) optional cGAN
        - update D on (real, fake)
        - update G to fool D
        - feed detached fake clips into the classifier as extra
          training samples for the fine-grained classes

The "ablation" knobs are loss weights in the yaml -- set any of
loss.subj_adv / loss.gan / loss.pgd to 0 to drop that piece.
"""
import argparse, os, json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import load_yaml, pick_device, set_seed, Meter, ensure_dir
from .data_loader import SoliDataset, SyntheticSoli, load_split
from .model import build_advsoli
from .losses import cls_loss, gan_d_loss, gan_g_loss, dann_lambda
from .adversarial import pgd_attack


def make_loaders(cfg, fold, splits_path, use_synth):
    if use_synth:
        # for sanity/quick mode -- skip real data
        train_ds = SyntheticSoli(n=120, seq_len=cfg['data']['seq_len'],
                                 num_classes=cfg['data']['num_classes'],
                                 num_subjects=cfg['data']['num_subjects'], seed=11)
        test_ds = SyntheticSoli(n=40, seq_len=cfg['data']['seq_len'],
                                num_classes=cfg['data']['num_classes'],
                                num_subjects=cfg['data']['num_subjects'], seed=99)
    else:
        tr_sess, te_sess, sess2subj = load_split(splits_path, fold)
        common = dict(seq_len=cfg['data']['seq_len'],
                      channels=tuple(cfg['data']['channels']),
                      sess2subj=sess2subj, normalize=True)
        train_ds = SoliDataset(cfg['data']['root'], tr_sess, **common)
        test_ds  = SoliDataset(cfg['data']['root'], te_sess, **common)
    bs = cfg['train']['batch_size']
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=0, drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                          num_workers=0)
    return train_dl, test_dl


def evaluate(enc, cls, loader, device):
    enc.eval(); cls.eval()
    correct = total = 0
    fg_correct = fg_total = 0
    fine = {0, 1, 2, 3}
    per_class_c = [0] * 11; per_class_t = [0] * 11
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y = y.to(device)
            logits = cls(enc(x))
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
            for cc in range(11):
                m = (y == cc)
                per_class_t[cc] += int(m.sum().item())
                per_class_c[cc] += int((pred[m] == y[m]).sum().item())
            mfg = torch.tensor([int(int(yy) in fine) for yy in y.cpu()],
                               dtype=torch.bool)
            fg_total   += int(mfg.sum().item())
            fg_correct += int((pred[mfg.to(device)] == y[mfg.to(device)]).sum().item())
    overall = correct / max(1, total)
    fg_acc  = fg_correct / max(1, fg_total)
    pc_acc  = [c / max(1, t) for c, t in zip(per_class_c, per_class_t)]
    return overall, fg_acc, pc_acc


def sample_fine_labels(batch_size, fine_classes, device):
    idx = torch.randint(0, len(fine_classes), (batch_size,))
    return torch.tensor([fine_classes[i] for i in idx], device=device)


def train_one_fold(cfg, fold, args):
    device = pick_device(cfg['train']['device'])
    set_seed(cfg['train']['seed'])
    print(f"[fold {fold}] device = {device}")

    use_synth = args.synthetic
    train_dl, test_dl = make_loaders(cfg, fold, args.splits, use_synth)
    print(f"[fold {fold}] train batches: {len(train_dl)}  test batches: {len(test_dl)}")

    enc, cls, subj, gen, disc = build_advsoli(
        cfg, num_classes=cfg['data']['num_classes'],
        num_subjects=cfg['data']['num_subjects'])
    enc, cls, subj = enc.to(device), cls.to(device), subj.to(device)
    gen, disc = gen.to(device), disc.to(device)

    lw = cfg['loss']
    use_dann = lw['subj_adv'] > 0
    use_gan  = lw['gan'] > 0
    use_pgd  = lw['pgd'] > 0
    fine = list(cfg['adversarial']['fine_grained_classes'])

    opt_main = torch.optim.Adam(
        list(enc.parameters()) + list(cls.parameters()) + list(subj.parameters()),
        lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    opt_g = torch.optim.Adam(gen.parameters(),  lr=cfg['train']['lr'], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg['train']['lr'], betas=(0.5, 0.999))

    epochs = cfg['train']['epochs'] if not args.quick else 3
    best_acc = -1.0
    ckpt_path = os.path.join(ensure_dir('checkpoints'),
                             f"fold{fold}_best.pt")
    log_path = os.path.join(ensure_dir('runs'), f"fold{fold}_log.json")
    history = []

    n_iters_total = max(1, epochs * len(train_dl))
    step = 0
    gan_warmup = max(1, epochs // 6)   # don't use fakes for cls until G is warm

    for ep in range(epochs):
        enc.train(); cls.train(); subj.train(); gen.train(); disc.train()
        m_cls, m_pgd, m_subj, m_gd, m_gg = (Meter() for _ in range(5))
        t0 = time.time()

        for x, y, s in train_dl:
            x = x.to(device); y = y.to(device); s = s.to(device)
            B = x.size(0)
            progress = step / n_iters_total
            alpha = dann_lambda(progress) if use_dann else 0.0

            # ---- main update (cls + DANN + PGD + GAN-aug) ----
            opt_main.zero_grad()

            feat = enc(x)
            logits = cls(feat)
            L = cls_loss(logits, y); L_cls_v = L.item()

            if use_pgd:
                fwd = lambda xx: cls(enc(xx))
                x_adv = pgd_attack(fwd, x, y,
                                   eps=cfg['adversarial']['pgd_eps'],
                                   alpha=cfg['adversarial']['pgd_alpha'],
                                   iters=cfg['adversarial']['pgd_iters'])
                logits_adv = cls(enc(x_adv))
                L_pgd = cls_loss(logits_adv, y)
                L = L + lw['pgd'] * L_pgd
                m_pgd.add(L_pgd.item())

            if use_dann:
                s_logits = subj(feat, alpha=alpha)
                L_subj = cls_loss(s_logits, s)
                L = L + lw['subj_adv'] * L_subj
                m_subj.add(L_subj.item())

            # GAN-augmented classifier samples (only after warmup)
            if use_gan and ep >= gan_warmup:
                with torch.no_grad():
                    z = torch.randn(B, cfg['model']['noise_dim'], device=device)
                    y_fake = sample_fine_labels(B, fine, device)
                    x_fake = gen(z, y_fake)
                logits_fake = cls(enc(x_fake))
                L_aug = cls_loss(logits_fake, y_fake)
                L = L + lw['gan'] * L_aug

            L.backward()
            opt_main.step()
            m_cls.add(L_cls_v)

            # ---- discriminator update ----
            if use_gan:
                opt_d.zero_grad()
                z = torch.randn(B, cfg['model']['noise_dim'], device=device)
                y_fake = sample_fine_labels(B, fine, device)
                with torch.no_grad():
                    x_fake = gen(z, y_fake)
                # use real samples whose class is in `fine` if possible,
                # otherwise use the whole batch (just label-conditioned)
                d_real = disc(x, y)
                d_fake = disc(x_fake, y_fake)
                Ld = gan_d_loss(d_real, d_fake)
                Ld.backward()
                opt_d.step()
                m_gd.add(Ld.item())

                # ---- generator update ----
                opt_g.zero_grad()
                z = torch.randn(B, cfg['model']['noise_dim'], device=device)
                y_fake = sample_fine_labels(B, fine, device)
                x_fake = gen(z, y_fake)
                d_fake_for_g = disc(x_fake, y_fake)
                Lg = gan_g_loss(d_fake_for_g)
                Lg.backward()
                opt_g.step()
                m_gg.add(Lg.item())

            step += 1

        ovr, fgacc, pc = evaluate(enc, cls, test_dl, device)
        rec = {
            "epoch": ep, "test_acc": ovr, "fg_acc": fgacc,
            "per_class": pc,
            "L_cls": m_cls.mean, "L_pgd": m_pgd.mean,
            "L_subj": m_subj.mean, "L_d": m_gd.mean, "L_g": m_gg.mean,
            "alpha_dann": alpha, "secs": time.time() - t0,
        }
        history.append(rec)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        is_best = ovr > best_acc
        if is_best:
            best_acc = ovr
            torch.save({
                "encoder": enc.state_dict(),
                "classifier": cls.state_dict(),
                "subject": subj.state_dict(),
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "config": cfg, "fold": fold, "epoch": ep,
                "best_acc": best_acc, "fg_acc": fgacc,
            }, ckpt_path)
        print(f"[fold {fold}] ep {ep:02d}  acc {ovr:.3f}  fg {fgacc:.3f}  "
              f"L_cls {m_cls.mean:.3f}  L_subj {m_subj.mean:.3f}  "
              f"L_d {m_gd.mean:.3f}  L_g {m_gg.mean:.3f}  "
              f"alpha {alpha:.2f}  ({rec['secs']:.1f}s)"
              + ("  *best*" if is_best else ""))

    return best_acc, ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fold", type=int, default=None,
                    help="0 or 1; if omitted, runs both")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--quick", action="store_true",
                    help="3 epochs, useful for sanity checking")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on the synthetic fallback dataset")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    folds = [args.fold] if args.fold is not None else [0, 1]
    results = {}
    for f in folds:
        acc, ckpt = train_one_fold(cfg, f, args)
        results[f"fold{f}"] = {"best_acc": acc, "ckpt": ckpt}

    print("\n=== summary ===")
    for k, v in results.items():
        print(k, "->", v)


if __name__ == "__main__":
    main()
