"""
Two adversarial pieces used in AdvSoli:

1. GradReverse / grad_reverse  -- the classic Ganin-Lempitsky GRL used by DANN.
   Forward is identity, backward multiplies the gradient by -alpha.
   This is what turns a normal classification head into an "adversary"
   without any tricks in the optimizer.

2. pgd_attack  -- L_inf projected gradient descent attack on the input.
   Used during training to push the encoder towards features that are
   robust to small radar-domain perturbations (Madry et al. 2018).
"""
import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.alpha * grad_out, None


def grad_reverse(x, alpha=1.0):
    return _GradReverse.apply(x, alpha)


def pgd_attack(forward_fn, x, y, eps=0.03, alpha=0.01, iters=3,
               loss_fn=None, clamp=(0.0, 1.0)):
    """
    Vanilla L_inf PGD.
    forward_fn: callable that takes x and returns logits over classes.
    Returns adversarial x with the same shape as x, detached from the graph.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # random init inside the eps ball
    x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(*clamp)

    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits = forward_fn(x_adv)
        loss = loss_fn(logits, y)
        g = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * g.sign()
            # project back into eps ball around x
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(*clamp)
    return x_adv.detach()
