"""Shared utilities for Question 2 (Motion or Memory?)."""
import numpy as np
import torch

DATA_ROOT = r"C:\Users\PC\Desktop\Artlist_ImageModel_Showdown\outputs\pa2_work\data"


def load_moving_mnist():
    """All 10,000 videos as float32 tensor (10000, 20, 1, 64, 64) in [0,1]."""
    import torchvision
    ds = torchvision.datasets.MovingMNIST(root=DATA_ROOT, split=None, download=True)
    v = ds.data if hasattr(ds, "data") else torch.stack([ds[i] for i in range(len(ds))])
    v = v.float() / 255.0
    if v.dim() == 4:  # (N, T, H, W) -> add channel
        v = v.unsqueeze(2)
    assert v.shape == (10000, 20, 1, 64, 64), v.shape
    return v


def split(videos):
    return videos[:8000], videos[8000:9000], videos[9000:10000]


# ---- transformations (applied to every frame; act on last two dims) ----
def M(x):
    return torch.flip(x, dims=(-1,))          # (Mx)[i,j] = x[i, 63-j]


def R(x):
    return torch.rot90(x, k=1, dims=(-2, -1))  # 90 deg anticlockwise


def Rpow(x, a):
    return torch.rot90(x, k=a % 4, dims=(-2, -1))


def g_apply(x, g):
    """g in 0..7: 0-3 -> R^a, 4-7 -> M R^a (rotate first, then mirror)."""
    a, m = g % 4, g >= 4
    y = Rpow(x, a)
    return M(y) if m else y


def g_inverse(x, g):
    a, m = g % 4, g >= 4
    if m:
        return Rpow(M(x), -a)   # (M R^a)^-1 = R^-a M
    return Rpow(x, -a)


# ---- windows ----
def window(video, t, k=1):
    """video (20,...) ; t 1-indexed start; returns X (5 frames) and Y (<=5 frames).
    X = (x_t, x_{t+k}, ..., x_{t+4k}), Y = (x_{t+5k}, ..., x_{t+9k}) keeping frame<=20."""
    xi = [t + j * k for j in range(5)]
    yi = [t + (5 + j) * k for j in range(5)]
    yi = [f for f in yi if f <= 20]
    return xi, yi


def window_frames(video, t, k=1):
    xi, yi = window(video, t, k)
    return video[[f - 1 for f in xi]], video[[f - 1 for f in yi]]


def allowed_starts(k, speed_aug):
    if not speed_aug:
        return list(range(1, 12))          # t in 1..11 at 1x
    # all five inputs and >=2 targets must exist: t + 6k <= 20
    return [t for t in range(1, 21) if t + 6 * k <= 20]
