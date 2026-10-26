"""ConvLSTM forecaster (model A) and larger-reach variant (model D)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid, k=3, dil=1):
        super().__init__()
        self.hid = hid
        pad = dil * (k - 1) // 2
        self.conv = nn.Conv2d(cin + hid, 4 * hid, k, padding=pad, dilation=dil, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], 1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Forecaster(nn.Module):
    """Model A with defaults.  Larger-reach designs only change kernel / dilation
    of the two ConvLSTM layers (same shapes, same 9-step rollout)."""

    def __init__(self, hid=64, ks=(3, 3), dils=(1, 1)):
        super().__init__()
        self.l1 = ConvLSTMCell(16, hid, ks[0], dils[0])
        self.l2 = ConvLSTMCell(hid, hid, ks[1], dils[1])
        self.out = nn.Conv2d(hid, 16, 1)
        self.hid = hid

    def step(self, frame, st1, st2):
        u = F.pixel_unshuffle(frame, 4)            # B,16,16,16
        h1, c1 = self.l1(u, st1)
        h2, c2 = self.l2(h1, st2)
        pred = torch.sigmoid(F.pixel_shuffle(self.out(h2), 4))  # B,1,64,64
        return pred, (h1, c1), (h2, c2)

    def forward(self, X):
        """X: (B,5,1,64,64) -> (B,5,1,64,64) = x̂_6..x̂_10; free-running rollout."""
        B = X.shape[0]
        z = X.new_zeros(B, self.hid, 16, 16)
        st1, st2 = (z, z.clone()), (z.clone(), z.clone())
        preds = []
        u = X[:, 0]
        for tau in range(1, 10):
            u = X[:, tau - 1] if tau <= 5 else preds[-1]
            p, st1, st2 = self.step(u, st1, st2)
            preds.append(p)
        return torch.stack(preds[4:], 1)


def build(name):
    """A/B/C: baseline.  D / DB: chosen larger-reach design (see choose_d.py)."""
    if name in ("A", "B", "C"):
        return Forecaster()
    import json, os
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "d_design.json")))
    return Forecaster(hid=cfg["hid"], ks=tuple(cfg["ks"]), dils=tuple(cfg["dils"]))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def reach_px(ks, dils, patch=4):
    """One-step reach in original pixels (centre-to-centre), per the handout conventions.
    Per-conv radius in folded cells = dil*(k-1)/2.
      from u_t   : layer-1 input conv, then layer-2 input conv
      from h1_t-1: layer-1 recurrent conv, then layer-2 input conv
      from h2_t-1: layer-2 recurrent conv
    (The 1x1 output conv, pixel_(un)shuffle add nothing.)"""
    r = [d * (k - 1) // 2 for k, d in zip(ks, dils)]
    cells = max(r[0] + r[1], r[0] + r[1], r[1])
    return cells * patch
