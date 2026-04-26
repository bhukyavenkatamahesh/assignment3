"""shared utilities: device selection, seeding, simple averaging meter."""
import os, random, yaml
import numpy as np
import torch


def pick_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


class Meter:
    """tracks running mean of a scalar; not thread-safe."""
    def __init__(self):
        self.n = 0
        self.s = 0.0
    def add(self, val, k=1):
        self.s += float(val) * k
        self.n += k
    @property
    def mean(self):
        return self.s / max(1, self.n)
    def reset(self):
        self.n = 0; self.s = 0.0


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
