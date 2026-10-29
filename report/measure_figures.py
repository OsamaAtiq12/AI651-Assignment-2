"""Measure the claims made about Figures 2, 3, 5 and 6a. Writes the three _measured*.json files.

Run from q2/ with the trained weights in ckpt/ (they are not in the repository):
    python ../report/measure_figures.py

What it reports:
  _measured.json   video 120 and 74 digit counts, centroid drift, prediction sharpness,
                   and the row-3 vs row-4 difference for model A and the final model
  _measured2.json  dilation-based digit counts for video 120 and 74, and one-epoch timings for D and D+B
  _measured3.json  digit counts and peak intensities for the four Figure 3 videos
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q2"))
import common as C
import evalib as E
import train as T
from common import M


def blobs(frame, thr=0.2):
    m = frame > thr
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = sorted((int(x) for x in np.bincount(lab.ravel())[1:]), reverse=True)
    return n, sizes[:2]


def digits(frame, thr=0.2, iters=2):
    """Separate digits: threshold, dilate so one digit's own strokes merge, then label."""
    m = frame > thr
    m = ndimage.binary_dilation(m, np.ones((3, 3)), iterations=iters)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = sorted((int(x) for x in np.bincount(lab.ravel())[1:]), reverse=True)
    return n, sizes[:3]


def centroid(frame, thr=0.2):
    ys, xs = np.nonzero(frame > thr)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def spread(P):
    vals = P[0] if P.dim() == 4 else P
    w = vals.sum(dim=0).cpu().numpy()
    tot = w.sum()
    if tot <= 0:
        return None
    ys, xs = np.mgrid[0 : w.shape[0], 0 : w.shape[1]]
    cx = (w * xs).sum() / tot
    cy = (w * ys).sum() / tot
    return float(np.sqrt((w * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / tot))


out = {}
test = E.test_videos()
X, Y = E.windows(test)
X, Y = X.to(E.DEV), Y.to(E.DEV)
A = E.load_model("A", 0)
final = E.Sym(E.load_model("DB", 0))
PA = E.predict(A, X)

# ---- video 120 (Figure 2)
out["video120_truth_blobs"] = [
    {"frame": f + 1, "blobs": blobs(test[120, f, 0].cpu().numpy())[0],
     "largest": blobs(test[120, f, 0].cpu().numpy())[1]}
    for f in range(20)
]
out["video120_truth_frames_separate"] = [
    r["frame"] for r in out["video120_truth_blobs"] if r["blobs"] >= 2
]
out["video120_prediction_blobs"] = [
    {"horizon": h + 1, "blobs": blobs(PA[120, h, 0].cpu().numpy())[0],
     "largest": blobs(PA[120, h, 0].cpu().numpy())[1]}
    for h in range(5)
]
out["video120_prediction_sharpness"] = [float(PA[120, h].max()) for h in range(5)]
out["video120_truth_sharpness"] = [float(Y[120, h].max()) for h in range(5)]

# ---- video 74 (Figure 5)
out["video74"] = {}
for k in (1, 2, 3):
    Xk, Yk = E.windows(test, k=k)
    Xk, Yk = Xk.to(E.DEV), Yk.to(E.DEV)
    Pk = E.predict(A, Xk)
    entry = {
        "input_blob_counts": [blobs(Xk[74, f, 0].cpu().numpy())[0] for f in range(5)],
        "truth_blob_counts_h1_h2": [blobs(Yk[74, h, 0].cpu().numpy())[0] for h in range(2)],
        "prediction_to_truth_centroid_px": [],
        "prediction_spread_px_h1_h2": [spread(Pk[74, h]) for h in range(2)],
        "prediction_max_h1_h2": [float(Pk[74, h].max()) for h in range(2)],
        "truth_max_h1_h2": [float(Yk[74, h].max()) for h in range(2)],
    }
    for h in range(2):
        cp = centroid(Pk[74, h, 0].cpu().numpy())
        ct = centroid(Yk[74, h, 0].cpu().numpy())
        entry["prediction_to_truth_centroid_px"].append(
            None if cp is None or ct is None else float(np.hypot(cp[0] - ct[0], cp[1] - ct[1]))
        )
    out["video74"][f"{k}x"] = entry

# ---- rows 3 vs 4 in Figure 3 / 6a
out["rows3_vs_4"] = {
    "A_video980_M": float(((E.predict(A, M(X)) - M(PA)) ** 2)[980].mean()),
    "final_video980_M": float(((E.predict(final, M(X)) - M(E.predict(final, X))) ** 2)[980].mean()),
    "A_mean_gap_M": float(((E.predict(A, M(X)) - M(PA)) ** 2).mean()),
    "final_mean_gap_M": float(((E.predict(final, M(X)) - M(E.predict(final, X))) ** 2).mean()),
}
out["sharpness"] = {
    "A_video980_max": float(PA[980].max()),
    "final_video980_max": float(E.predict(final, X)[980].max()),
    "A_video254_max": float(PA[254].max()),
    "final_video254_max": float(E.predict(final, X)[254].max()),
}
Path(os.path.join(os.path.dirname(__file__), "_measured.json")).write_text(
    json.dumps(out, indent=1), encoding="utf8"
)

# ---- dilation-based digit counts and one-epoch timings
vids = C.load_moving_mnist()
out2 = {"overlap": {}, "timing": {}}
for name, idx in (("video120", 9120), ("video74", 9074)):
    out2["overlap"][name] = [
        {"frame": f + 1, "digits": digits(vids[idx, f, 0].numpy())[0],
         "sizes": digits(vids[idx, f, 0].numpy())[1]}
        for f in range(20)
    ]
for name in ("D", "DB"):
    T.SPEED_AUG[name], T.SYM_AUG[name] = True, name == "DB"
    t0 = time.time()
    T.train(name, 0, epochs=1, out_dir="ckpt_timing", quiet=True)
    out2["timing"][name] = time.time() - t0
Path(os.path.join(os.path.dirname(__file__), "_measured2.json")).write_text(
    json.dumps(out2, indent=1), encoding="utf8"
)

# ---- Figure 3 videos
PF = E.predict(final, X)
out3 = {}
for idx, tag in ((673, "video673_M"), (980, "video980_M"), (254, "video254_R"), (982, "video982_R")):
    out3[tag] = {
        "truth_digits_frames": [digits(test[idx, f, 0].cpu().numpy())[0] for f in range(10)],
        "truth_digits_future_sizes": [digits(Y[idx, h, 0].cpu().numpy())[1][:2] for h in range(5)],
        "A_pred_digits": [digits(PA[idx, h, 0].cpu().numpy())[0] for h in range(5)],
        "F_pred_digits": [digits(PF[idx, h, 0].cpu().numpy())[0] for h in range(5)],
        "truth_peak": [float(Y[idx, h].max()) for h in range(5)],
        "A_peak": [float(PA[idx, h].max()) for h in range(5)],
        "F_peak": [float(PF[idx, h].max()) for h in range(5)],
        "A_mass": [float(PA[idx, h].sum()) for h in range(5)],
        "truth_mass": [float(Y[idx, h].sum()) for h in range(5)],
    }
Path(os.path.join(os.path.dirname(__file__), "_measured3.json")).write_text(
    json.dumps(out3, indent=1), encoding="utf8"
)
print("wrote _measured.json, _measured2.json, _measured3.json")
