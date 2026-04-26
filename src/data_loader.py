"""
Soli range-Doppler data loader.

Each .h5 has datasets ch0..ch3 (radar channels) of shape [T, 1024]
which we reshape to [T, 32, 32], plus a 'label' dataset (per-frame labels;
we just take the majority class for the clip).

Returned sample:
    rdm:     torch.FloatTensor of shape [seq_len, 4, 32, 32]   (normalized to [0,1])
    label:   int, gesture class
    subj:    int, subject index
"""
import os, glob, json, re
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


_FNAME_RE = re.compile(r"(\d+)_(\d+)_(\d+)\.h5")


def _parse_name(path):
    m = _FNAME_RE.search(os.path.basename(path))
    if m is None:
        return None
    return tuple(int(x) for x in m.groups())  # (gesture, session, instance)


def _pad_or_crop(arr, T):
    # arr is [t, C, H, W]. Pad with zeros at the end or crop center.
    t = arr.shape[0]
    if t == T:
        return arr
    if t > T:
        start = (t - T) // 2
        return arr[start:start + T]
    pad = np.zeros((T - t, *arr.shape[1:]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


class SoliDataset(Dataset):
    """One sample per .h5 file."""

    def __init__(self, root, sessions, seq_len=32, channels=(0, 1, 2, 3),
                 sess2subj=None, normalize=True, in_memory=False):
        self.root = root
        self.seq_len = seq_len
        self.channels = list(channels)
        self.normalize = normalize
        self.sess2subj = sess2subj or {}
        self.in_memory = in_memory

        all_files = sorted(glob.glob(os.path.join(root, "*.h5")))
        keep = []
        for f in all_files:
            parsed = _parse_name(f)
            if parsed is None:
                continue
            g, s, i = parsed
            if s in sessions:
                keep.append((f, g, s, i))
        if not keep:
            raise RuntimeError(f"no samples in {root} for sessions {sessions}")
        self.items = keep

        # caching the whole thing in RAM is feasible (~few GB after subsetting)
        self._cache = {}
        if self.in_memory:
            for idx in range(len(self.items)):
                self._cache[idx] = self._load(idx)

    def __len__(self):
        return len(self.items)

    def _load(self, idx):
        path, gesture, sess, _ = self.items[idx]
        with h5py.File(path, "r") as h:
            chans = []
            for c in self.channels:
                key = f"ch{c}"
                if key not in h:
                    # missing channel, fill with zeros to keep shape stable
                    chans.append(np.zeros_like(np.asarray(h[f"ch{self.channels[0]}"])))
                else:
                    chans.append(np.asarray(h[key], dtype=np.float32))
            # each c: [T, 1024]
            T = min(c.shape[0] for c in chans)
            chans = [c[:T].reshape(T, 32, 32) for c in chans]
            x = np.stack(chans, axis=1)  # [T, C, 32, 32]
            # gesture from filename is reliable; sometimes 'label' has -1 frames
            label = gesture
        return x, label, sess

    def __getitem__(self, idx):
        if idx in self._cache:
            x, label, sess = self._cache[idx]
        else:
            x, label, sess = self._load(idx)

        x = _pad_or_crop(x, self.seq_len)
        if self.normalize:
            # min-max per clip; soli RDMs are non-negative magnitudes
            mn, mx = x.min(), x.max()
            if mx > mn:
                x = (x - mn) / (mx - mn)
        x = torch.from_numpy(x.astype(np.float32))
        subj = self.sess2subj.get(int(sess), int(sess))
        return x, int(label), int(subj)


def load_split(splits_path, fold):
    with open(splits_path) as f:
        sp = json.load(f)
    sess2subj = {int(k): v for k, v in sp["session_to_subject"].items()}
    fold_cfg = sp["folds"][fold]
    return fold_cfg["train_sessions"], fold_cfg["test_sessions"], sess2subj


# ---------------------------------------------------------------------------
# tiny synthetic dataset, used for sanity-testing the rest of the pipeline
# before the 10G download is ready.
# ---------------------------------------------------------------------------
class SyntheticSoli(Dataset):
    """Random RDMs with a class-conditional motion blob, just for code paths."""

    def __init__(self, n=200, seq_len=32, num_classes=11, num_subjects=10, seed=0):
        rng = np.random.default_rng(seed)
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.samples = []
        for _ in range(n):
            cls = int(rng.integers(0, num_classes))
            subj = int(rng.integers(0, num_subjects))
            # base noise
            x = rng.normal(0.05, 0.02, size=(seq_len, 4, 32, 32)).astype(np.float32)
            # add a class-conditional moving blob
            cy = 8 + (cls % 4) * 4
            cx = 8 + (cls // 4) * 4
            for t in range(seq_len):
                yy = (cy + t // 4) % 32
                xx = (cx + (t * (cls + 1)) // 6) % 32
                x[t, :, max(0, yy-1):yy+2, max(0, xx-1):xx+2] += 0.6
            x = np.clip(x, 0, 1)
            self.samples.append((x, cls, subj))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, cls, subj = self.samples[i]
        return torch.from_numpy(x), cls, subj
