"""
Loss helpers for AdvSoli, all written from scratch
(no torchvision / no pre-built combined losses).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cls_loss(logits, target):
    # cross entropy, mean over batch
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return nll.mean()


def bce_with_logits(pred, target):
    # we re-implement to keep things explicit (no nn.BCEWithLogits)
    # numerically stable form
    z = pred
    t = target
    return (torch.clamp(z, min=0) - z * t + torch.log1p(torch.exp(-z.abs()))).mean()


def gan_d_loss(d_real_logit, d_fake_logit):
    # standard non-saturating GAN: real -> 1, fake -> 0
    real_t = torch.ones_like(d_real_logit)
    fake_t = torch.zeros_like(d_fake_logit)
    return 0.5 * (bce_with_logits(d_real_logit, real_t) +
                  bce_with_logits(d_fake_logit, fake_t))


def gan_g_loss(d_fake_logit):
    # generator wants D to think fakes are real
    real_t = torch.ones_like(d_fake_logit)
    return bce_with_logits(d_fake_logit, real_t)


def dann_lambda(progress, gamma=10.0):
    """
    Schedule used in the original DANN paper.
    progress in [0,1] across training. Smoothly grows from 0 to 1.
    """
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * progress))) - 1.0)
