"""Table 1 (sanity checks) and Figure 1 (optical-flow statistics of training videos)."""
import sys, json
import numpy as np, torch, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

out = {}
torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64)
p[7, 0, 10, 20] = 1

MRM = lambda z: g_apply(R(g_apply(z, 4)), 4)  # placeholder, replaced below
# (c) MRM(x) with M R M: mirror, rotate, mirror -> equals R^-1
MRM = lambda z: M(R(M(z)))

out["a"] = (M(M(x)) - x).abs().max().item()
out["b"] = (R(R(R(R(x)))) - x).abs().max().item()
out["c"] = (MRM(x) - Rpow(x, -1)).abs().max().item()
out["c_equal"] = bool(torch.equal(MRM(x), Rpow(x, -1)))
out["a_equal"] = bool(torch.equal(M(M(x)), x))
out["b_equal"] = bool(torch.equal(R(R(R(R(x)))), x))


def bright(v):
    idx = (v == 1).nonzero()
    return tuple(idx[0][[0, 2, 3]].tolist())


out["d"] = bright(M(p))
out["e"] = bright(R(p))

v = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).contiguous()
for name, k in (("f", 1), ("g", 2), ("h", 3)):
    X, Y = window_frames(v, 1, k)
    out[name] = ([int(a) for a in X[:, 0, 0, 0]], [int(a) for a in Y[:, 0, 0, 0]])
print(json.dumps(out, indent=1))
json.dump(out, open("table1.json", "w"), indent=1)

# ---------------- Figure 1 ----------------
vids = load_moving_mnist()[:50]
u8 = (vids[:, :, 0].numpy() * 255).astype(np.uint8)
thetas, speeds = [], []
for n in range(50):
    for s in range(19):
        prev, nxt = u8[n, s], u8[n, s + 1]
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(dx ** 2 + dy ** 2)
        keep = (prev > 51) & (nxt > 51) & (mag > 0.5)
        thetas.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360)
        speeds.append(mag[keep])
theta = np.concatenate(thetas); speed = np.concatenate(speeds)
med = float(np.median(speed)); npx = int(len(speed))
print("kept pixels", npx, "median speed", med)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
w = np.ones_like(theta) / len(theta)
ax[0].hist(theta, bins=18, range=(0, 360), weights=w, color="#4a6fa5", edgecolor="w")
ax[0].axhline(1 / 18, color="r", ls="--", label="uniform (1/18)")
ax[0].set_xlabel("flow direction θ (deg; 0° = right, 90° = up)"); ax[0].set_ylabel("fraction of kept pixels")
ax[0].set_xticks(range(0, 361, 60)); ax[0].legend(); ax[0].set_title("(a) direction")
ws = np.ones_like(speed) / len(speed)
ax[1].hist(speed, bins=40, weights=ws, color="#4a6fa5", edgecolor="w")
ax[1].axvline(med, color="r", label=f"median = {med:.3g}")
ax[1].set_xlabel("flow speed (pixels / frame)"); ax[1].set_ylabel("fraction of kept pixels")
ax[1].legend(); ax[1].set_title("(b) speed")
fig.suptitle(f"Figure 1: optical flow, first 50 training videos (950 pairs); median speed {med:.3g} px/frame; {npx:,} pixels kept")
plt.tight_layout(); plt.savefig("figure1.png", dpi=150)
hist, _ = np.histogram(theta, bins=18, range=(0, 360))
json.dump({"median_speed": med, "n_kept": npx, "dir_fractions": (hist / hist.sum()).tolist(),
           "speed_pcts": np.percentile(speed, [5, 25, 50, 75, 95]).tolist()}, open("figure1.json", "w"), indent=1)
