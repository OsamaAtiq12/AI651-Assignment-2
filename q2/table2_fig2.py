"""Table 2 (model A vs Blank / Copy-last) and Figure 2 (qualitative examples)."""
import json
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evalib import *

res = {}
for s in (0, 1):
    f = load_model("A", s)
    h = horizon_mse(f)
    res[f"A_s{s}"] = {"mse": h.tolist(), "E5": float(h.mean()), "params": n_params(f)}
for nm, f in (("Blank", Blank()), ("Copy-last", CopyLast())):
    h = horizon_mse(f); res[nm] = {"mse": h.tolist(), "E5": float(h.mean()), "params": 0}
a = np.mean([res["A_s0"]["mse"], res["A_s1"]["mse"]], 0)
res["A_mean"] = {"mse": a.tolist(), "E5": float(a.mean())}
json.dump(res, open("table2.json", "w"), indent=1)
for k, v in res.items(): print(k, ["%.4g" % x for x in v["mse"]], "E5 %.4g" % v["E5"], v.get("params"))

# ---- Figure 2: three test videos, >=1 with overlapping digits ----
f = load_model("A", 0)
vids = test_videos()
X, Y = windows(vids)
P = predict(f, X)
# overlap score: how many pixels of the first/last target are bright where two digits cross -> use
# intensity>1.0 proxy is impossible (clipped); instead use the per-video error growth: pick the
# worst error video as an overlap candidate, and verify visually.  Overlap proxy: pixels in the
# 5 target frames with intensity at the mean of overlapping digits (>0.99) after smoothing is not
# reliable, so we use "digit area drops sharply across a frame" (digits merging).
area = (Y > 0.3).float().sum(dim=(2, 3, 4))                 # (N,5)
overlap_score = (area[:, 0] - area.min(1).values).cpu().numpy()
err = ((P - Y) ** 2).mean(dim=(1, 2, 3, 4)).cpu().numpy()
ov_idx = int(np.argsort(-overlap_score)[0])
order = np.argsort(err)
picks = [ov_idx, int(order[len(order) // 4]), int(order[len(order) // 2])]
picks = list(dict.fromkeys(picks))[:3]
json.dump({"picks": picks, "ov": ov_idx}, open("fig2_picks.json", "w"))
fig, axs = plt.subplots(3 * len(picks), 10, figsize=(13, 1.35 * 3 * len(picks)))
for a_ in axs.ravel(): a_.axis("off")
for r, n in enumerate(picks):
    for c in range(5):
        axs[3 * r, c].imshow(X[n, c, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axs[3 * r + 1, 5 + c].imshow(P[n, c, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axs[3 * r + 2, 5 + c].imshow(Y[n, c, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    axs[3 * r, 0].set_title(f"test video #{n} (dataset idx {9000 + n}): row1 input x1-5, row2 prediction x6-10, row3 truth x6-10", fontsize=8, loc="left")
plt.tight_layout(); plt.savefig("figure2.png", dpi=140)
print("picked", picks)
