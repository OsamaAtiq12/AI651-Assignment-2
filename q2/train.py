"""Train one model: python train.py --model A --seed 0  (protocol of the handout)."""
import argparse, json, os, time
import torch
from common import *
from models import build, n_params

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPEED_AUG = {"A": False, "B": False, "C": True, "D": True, "DB": True}
SYM_AUG = {"A": False, "B": True, "C": False, "D": False, "DB": True}


def make_batch(vids, idx, rng, speed_aug, sym_aug):
    """Build one batch: X (B,5,1,64,64), Y (B,5,1,64,64) zero-padded, mask (B,5)."""
    X, Y, Mk = [], [], []
    for n in idx.tolist():
        k = int(torch.randint(1, 4, (1,), generator=rng)) if speed_aug else 1
        starts = allowed_starts(k, speed_aug)
        t = starts[int(torch.randint(0, len(starts), (1,), generator=rng))]
        x, y = window_frames(vids[n], t, k)
        if sym_aug:
            g = int(torch.randint(0, 8, (1,), generator=rng))
            x, y = g_apply(x, g), g_apply(y, g)
        m = torch.zeros(5, device=vids.device)
        m[: y.shape[0]] = 1
        if y.shape[0] < 5:
            y = torch.cat([y, torch.zeros(5 - y.shape[0], *y.shape[1:], device=y.device)])
        X.append(x); Y.append(y); Mk.append(m)
    return torch.stack(X), torch.stack(Y), torch.stack(Mk)


def masked_loss(pred, Y, mask):
    se = ((pred - Y) ** 2).mean(dim=(2, 3, 4))          # (B,5) per-frame MSE
    per_video = (se * mask).sum(1) / mask.sum(1)
    return per_video.mean()


@torch.no_grad()
def val_loss(model, vids, bs=250):
    """Validation E^(5) at 1x, t=1 (monitoring / design choice only)."""
    model.eval(); tot = 0.0
    for i in range(0, len(vids), bs):
        X, Y = vids[i:i + bs, :5], vids[i:i + bs, 5:10]
        tot += ((model(X) - Y) ** 2).mean(dim=(2, 3, 4)).mean(1).sum().item()
    model.train()
    return tot / len(vids)


def train(name, seed, epochs=30, out_dir="ckpt", quiet=False, model=None, tag=None):
    os.makedirs(out_dir, exist_ok=True)
    allv = load_moving_mnist()
    tr, va, _ = split(allv)
    tr, va = tr.to(DEV), va.to(DEV)
    torch.manual_seed(seed)
    model = (model or build(name)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = torch.Generator().manual_seed(seed)
    log = {"model": name, "seed": seed, "params": n_params(model), "device": DEV,
           "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "epochs": []}
    for ep in range(1, epochs + 1):
        if DEV == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        perm = torch.randperm(len(tr), generator=rng)
        run = 0.0
        for i in range(0, len(tr), 32):
            X, Y, Mk = make_batch(tr, perm[i:i + 32], rng, SPEED_AUG[name], SYM_AUG[name])
            loss = masked_loss(model(X), Y, Mk)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * X.shape[0]
        if DEV == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0
        vl = val_loss(model, va)
        log["epochs"].append({"epoch": ep, "train_loss": run / len(tr), "val_E5": vl, "seconds": dt})
        if not quiet: print(f"{name} s{seed} ep{ep:02d} train {run/len(tr):.5f} val {vl:.5f} {dt:.1f}s", flush=True)
    log["epoch_time_mean"] = sum(e["seconds"] for e in log["epochs"]) / epochs
    tag = tag or f"{name}_s{seed}"
    torch.save(model.state_dict(), f"{out_dir}/{tag}.pt")
    json.dump(log, open(f"{out_dir}/{tag}.json", "w"), indent=1)
    return model, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--out", default="ckpt")
    a = ap.parse_args()
    train(a.model, a.seed, a.epochs, a.out)
