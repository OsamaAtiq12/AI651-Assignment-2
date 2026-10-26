"""Evaluation helpers shared by all table/figure scripts."""
import time, json
import numpy as np, torch
from common import *
from models import build, n_params

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_test = None


def test_videos():
    global _test
    if _test is None:
        _test = split(load_moving_mnist())[2].to(DEV)   # (1000,20,1,64,64)
    return _test


def load_model(name, seed, ckpt="ckpt", tag=None):
    m = build("D" if name == "DB" else name).to(DEV)
    m.load_state_dict(torch.load(f"{ckpt}/{tag or f'{name}_s{seed}'}.pt", map_location=DEV))
    return m.eval()


def windows(vids, k=1, t=1):
    xi, yi = window(None, t, k)
    X = vids[:, [f - 1 for f in xi]]
    Y = vids[:, [f - 1 for f in yi]]
    return X, Y


class Blank:
    def __call__(self, X): return torch.zeros_like(X)
    def eval(self): return self


class CopyLast:
    def __call__(self, X): return X[:, -1:].expand_as(X).contiguous()
    def eval(self): return self


class Sym:
    """Symmetrised forecaster, eq. (7)."""
    def __init__(self, f): self.f = f
    def eval(self): return self
    def __call__(self, X):
        acc = 0
        for g in range(8):
            acc = acc + g_inverse(self.f(g_apply(X, g)), g)
        return acc / 8


@torch.no_grad()
def predict(f, X, bs=250):
    return torch.cat([f(X[i:i + bs]) for i in range(0, len(X), bs)])


@torch.no_grad()
def horizon_mse(f, k=1, t=1):
    """(5 or fewer,) per-horizon MSE averaged over pixels and test videos."""
    X, Y = windows(test_videos(), k, t)
    P = predict(f, X)[:, : Y.shape[1]]
    return ((P - Y) ** 2).mean(dim=(0, 2, 3, 4)).cpu().numpy()


@torch.no_grad()
def E5(f, k=1): return float(horizon_mse(f, k).mean())


@torch.no_grad()
def E2(f, k=1): return float(horizon_mse(f, k)[:2].mean())


@torch.no_grad()
def per_video_CE(f, T):
    """Per-video consistency gap C_T and transformed error E_T at 1x, t=1."""
    X, Y = windows(test_videos())
    fX = predict(f, X)
    fTX = predict(f, T(X))
    C = ((fTX - T(fX)) ** 2).mean(dim=(1, 2, 3, 4))
    E = ((fTX - T(Y)) ** 2).mean(dim=(1, 2, 3, 4))
    return C.cpu().numpy(), E.cpu().numpy()


def CE(f, T):
    C, E = per_video_CE(f, T)
    return float(C.mean()), float(E.mean())


@torch.no_grad()
def inference_time(f, bs=32):
    vids = test_videos()
    X, _ = windows(vids)
    X = X.contiguous()
    best = []
    for rep in range(2):   # first pass warms up cuDNN; report the second
        if DEV == "cuda": torch.cuda.synchronize()
        tot = 0.0
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]
            if DEV == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            f(xb)
            if DEV == "cuda": torch.cuda.synchronize()
            tot += time.time() - t0
        best.append(tot)
    return best[-1]


def train_epoch_time(name, seed, ckpt="ckpt", tag=None):
    return json.load(open(f"{ckpt}/{tag or f'{name}_s{seed}'}.json"))["epoch_time_mean"]
