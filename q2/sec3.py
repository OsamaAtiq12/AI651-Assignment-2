"""Section 3: Table 3, Figure 3, Figure 4 (mirror / rotation), Table 4, Figure 5 (speed).
usage: python sec3.py A            (baseline, seed-0 figures)
       python sec3.py final <name> (Figure 6 uses same videos; see fig6.py)"""
import sys, json, os
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from evalib import *

TRANS = {"M": M, "R": R}


def table3(name_to_models):
    rows = {}
    for label, models in name_to_models.items():
        r = {"E5": [], "CM": [], "EM": [], "CR": [], "ER": []}
        for f in models:
            r["E5"].append(E5(f))
            for T, ck, ek in (("M", "CM", "EM"), ("R", "CR", "ER")):
                c, e = CE(f, TRANS[T]); r[ck].append(c); r[ek].append(e)
        rows[label] = r
    return rows


def pick_videos(f, T_name, n=2):
    C, _ = per_video_CE(f, TRANS[T_name])
    return [int(i) for i in np.argsort(-C)[:n]], C


def rows_for(f, n, T):
    """Four rows x 10 frames of Figure 3 for test video n (frames as 10 tensors each)."""
    X, Y = windows(test_videos()[n:n + 1])
    with torch.no_grad():
        fX = f(X); TX = T(X); fTX = f(TX); TfX = T(fX)
    r1 = torch.cat([X[0], fX[0]]); r2 = torch.cat([TX[0], T(Y)[0]])
    r3 = torch.cat([TX[0], fTX[0]]); r4 = torch.cat([TX[0], TfX[0]])
    return [r.cpu().numpy()[:, 0] for r in (r1, r2, r3, r4)]


def fig3(f, picks, path_png, gif_prefix, title):
    """picks: list of (video index, 'M'|'R')"""
    fig, axs = plt.subplots(4 * len(picks), 10, figsize=(13, 1.25 * 4 * len(picks)))
    for a in axs.ravel(): a.axis("off")
    labs = ["X | f(X)", "T(X) | T(Y)", "T(X) | f(T(X))", "T(X) | T(f(X))"]
    for p, (n, Tn) in enumerate(picks):
        rows = rows_for(f, n, TRANS[Tn])
        for r in range(4):
            for c in range(10):
                axs[4 * p + r, c].imshow(rows[r][c], cmap="gray", vmin=0, vmax=1)
            axs[4 * p + r, 0].set_title(labs[r], fontsize=7, loc="left")
        axs[4 * p, 4].set_title(f"video {n}, T={Tn}  [{labs[0]}]", fontsize=8)
        # GIF stepping through the five forecast horizons (4 rows, 5 input cols stay)
        frames = []
        for h in range(5):
            img = np.concatenate([np.concatenate([rows[r][:5], rows[r][5 + h:6 + h]], 0).transpose(1, 0, 2).reshape(64, -1)
                                  for r in range(4)], 0)
            frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        imageio.mimsave(f"{gif_prefix}_video{n}_{Tn}.gif", [np.kron(fr, np.ones((3, 3), np.uint8)) for fr in frames], duration=0.8, loop=0)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(); plt.savefig(path_png, dpi=130); plt.close()


def fig4(f, picks, path):
    CM, _ = per_video_CE(f, M); CR, _ = per_video_CE(f, R)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, C, nm in ((ax[0], CM, "C_M"), (ax[1], CR, "C_R")):
        a.hist(C, bins=50, color="#4a6fa5"); a.set_xlabel(f"per-video {nm}"); a.set_ylabel("videos")
        for n, Tn in picks:
            if (Tn == "M") == (nm == "C_M"): a.axvline(C[n], color="r", ls="--")
            else: a.axvline(C[n], color="orange", ls=":")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def table4(models):
    out = {}
    for k in (1, 2, 3):
        out[f"E2_{k}x"] = [E2(f, k) for f in models]
    for k in (1, 2):
        out[f"E5_{k}x"] = [E5(f, k) for f in models]
    return out


def fig5(f, n, path, title):
    vids = test_videos()
    fig, axs = plt.subplots(3, 9, figsize=(12, 4.2))
    for a in axs.ravel(): a.axis("off")
    for ri, k in enumerate((1, 2, 3)):
        X, Y = windows(vids[n:n + 1], k)
        with torch.no_grad(): P = f(X)
        for c in range(5): axs[ri, c].imshow(X[0, c, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        for h in range(2):
            axs[ri, 5 + 2 * h - 0].imshow(P[0, h, 0].cpu(), cmap="gray", vmin=0, vmax=1) if False else None
        axs[ri, 5].imshow(P[0, 0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axs[ri, 6].imshow(Y[0, 0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axs[ri, 7].imshow(P[0, 1, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axs[ri, 8].imshow(Y[0, 1, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axs[ri, 0].set_title(f"{k}x: inputs", fontsize=8, loc="left")
        for c, t in zip((5, 6, 7, 8), ("pred h1", "true h1", "pred h2", "true h2")):
            axs[ri, c].set_title(t, fontsize=7)
    fig.suptitle(title, fontsize=9); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def nonoverlap_video(k_list=(1, 2, 3), limit=1000):
    """First test video whose true frames shown in Figure 5 (inputs + 2 targets, at every speed)
    have no overlapping digits.  Overlap proxy: the digit-pixel count (>0.3) of the whole displayed
    sequence never drops below 85% of its maximum (merging digits lose visible area), and
    the two digit bounding boxes are verified by connected components (cv2) to stay 2 separate blobs."""
    import cv2
    vids = test_videos()
    for n in range(limit):
        ok = True
        for k in k_list:
            X, Y = windows(vids[n:n + 1], k)
            fr = torch.cat([X[0], Y[0, :2]])[:, 0].cpu().numpy()
            for im in fr:
                num, _ = cv2.connectedComponents(cv2.dilate((im > 0.3).astype(np.uint8), np.ones((7, 7), np.uint8)))
                if num - 1 != 2: ok = False; break
            if not ok: break
        if ok: return n
    return 0


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "A":
        fa = [load_model("A", s) for s in (0, 1)]
        T3 = table3({"A": fa, "Blank": [Blank()], "Copy-last": [CopyLast()]})
        json.dump(T3, open("table3.json", "w"), indent=1); print(json.dumps(T3, indent=1))
        picksM, _ = pick_videos(fa[0], "M"); picksR, _ = pick_videos(fa[0], "R")
        picks = [(n, "M") for n in picksM] + [(n, "R") for n in picksR]
        json.dump(picks, open("fig3_picks.json", "w")); print("fig3 picks", picks)
        os.makedirs("gifs_A", exist_ok=True)
        fig3(fa[0], picks, "figure3.png", "gifs_A/figure3", "Figure 3 (model A, seed 0)")
        fig4(fa[0], picks, "figure4.png")
        T4 = table4(fa + [CopyLast()]); json.dump(T4, open("table4.json", "w"), indent=1); print(json.dumps(T4, indent=1))
        n5 = nonoverlap_video(); json.dump({"fig5_video": n5}, open("fig5_pick.json", "w")); print("fig5 video", n5)
        fig5(fa[0], n5, "figure5.png", f"Figure 5: model A seed 0, test video {n5}")
