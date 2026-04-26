"""
AdvSoli model components.

  SoliEncoder        : per-frame 2D-CNN  +  Bi-GRU over time, returns a feature
  GestureClassifier  : small MLP head -> 11 classes
  SubjectHead        : same idea, with a GRL in front for DANN
  GestureGenerator   : conditional 3D-DeConv generator producing fake clips
  RFDiscriminator    : conditional 3D-CNN discriminator (real vs fake)

Notes:
- The encoder is intentionally small (~0.4M params) so it trains in MPS / CPU.
- I went with 2D-CNN-per-frame + GRU rather than full 3D-CNN because radar
  RDMs are very sparse; a 2D-CNN seems to extract per-frame "blobs" cleanly,
  and the GRU then ties the temporal motion together.
- GAN parts are kept minimal (no spectral norm, no progressive growing) since
  the goal is to demonstrate that adversarial augmentation helps the
  fine-grained classes, not to ship state-of-the-art image synthesis.
"""
import torch
import torch.nn as nn
from .adversarial import grad_reverse


def _conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2),
    )


class SoliEncoder(nn.Module):
    def __init__(self, in_ch=4, conv_channels=(16, 32, 64),
                 gru_hidden=128, feat_dim=128):
        super().__init__()
        layers = []
        prev = in_ch
        for c in conv_channels:
            layers.append(_conv_block(prev, c))
            prev = c
        self.cnn = nn.Sequential(*layers)
        # 32 -> 16 -> 8 -> 4 with three pools
        self.flat_dim = conv_channels[-1] * 4 * 4
        self.gru = nn.GRU(self.flat_dim, gru_hidden, batch_first=True,
                          bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden * 2, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T = x.shape[:2]
        h = x.reshape(B * T, *x.shape[2:])
        h = self.cnn(h)
        h = h.reshape(B, T, -1)
        out, _ = self.gru(h)
        feat = out.mean(dim=1)            # temporal mean pool
        return self.proj(feat)


class GestureClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SubjectHead(nn.Module):
    """Used for DANN-style adversarial subject discrimination."""

    def __init__(self, feat_dim, num_subjects, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_subjects),
        )

    def forward(self, x, alpha=1.0):
        return self.net(grad_reverse(x, alpha))


class GestureGenerator(nn.Module):
    """
    Conditional 3D-DeConv generator.
    Input:  noise z [B, noise_dim], class label [B] (long tensor)
    Output: fake clip [B, T=32, C=4, H=32, W=32] in [0,1]
    """
    def __init__(self, noise_dim=64, num_classes=11, embed_dim=16, base=32):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        in_dim = noise_dim + embed_dim
        self.start = 4
        self.base = base
        self.proj = nn.Linear(in_dim, base * self.start ** 3)

        self.up = nn.Sequential(
            nn.ConvTranspose3d(base,      base // 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(base // 2), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(base // 2, base // 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(base // 4), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(base // 4, base // 8, 4, stride=2, padding=1),
            nn.BatchNorm3d(base // 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Conv3d(base // 8, 4, 3, padding=1)

    def forward(self, z, y):
        h = torch.cat([z, self.embed(y)], dim=1)
        h = self.proj(h).view(-1, self.base, self.start, self.start, self.start)
        h = self.up(h)
        x = torch.sigmoid(self.out(h))            # [B, 4, T, H, W]
        return x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]


class RFDiscriminator(nn.Module):
    """Conditional real/fake discriminator on full clips."""

    def __init__(self, num_classes=11, embed_dim=16, base=16):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.cnn = nn.Sequential(
            nn.Conv3d(4, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base, base * 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(base * 4), nn.LeakyReLU(0.2, inplace=True),
        )
        # 32->16->8->4 in T, H, W. final feature: base*4 * 4*4*4
        feat = base * 4 * 4 * 4 * 4
        self.head = nn.Sequential(
            nn.Linear(feat + embed_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x, y):
        # x: [B, T, C, H, W] -> [B, C, T, H, W]
        h = self.cnn(x.permute(0, 2, 1, 3, 4))
        h = h.flatten(1)
        e = self.embed(y)
        return self.head(torch.cat([h, e], dim=1))


# small convenience to build the whole stack from a config dict
def build_advsoli(cfg, num_classes, num_subjects):
    m = cfg["model"]
    enc = SoliEncoder(in_ch=len(cfg["data"]["channels"]),
                      conv_channels=tuple(m["enc_channels"]),
                      gru_hidden=m["gru_hidden"],
                      feat_dim=m["feat_dim"])
    cls = GestureClassifier(m["feat_dim"], num_classes)
    subj = SubjectHead(m["feat_dim"], num_subjects)
    gen = GestureGenerator(noise_dim=m["noise_dim"], num_classes=num_classes)
    disc = RFDiscriminator(num_classes=num_classes)
    return enc, cls, subj, gen, disc
