import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 7.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")


def box(x, y, w, h, txt, fc="#dbe7f3", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec="#26496b", lw=1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


def arr(x1, y1, x2, y2, t="", fs=7, c="#26496b"):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=c, lw=1.1))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, t, fontsize=fs, ha="center", color="#7a2e2e")


# data sources (column 1)
box(0.1, 5.7, 2.2, 0.9, "Water levels S1-S4, T1\nL=36 steps x 5 stations\n(B, 36, 5) + mask + age", "#e9f1e4")
box(0.1, 4.5, 2.2, 0.9, "Rain gauges R1, R2\n(B, 36, 2) + mask", "#e9f1e4")
box(0.1, 3.3, 2.2, 0.9, "Past releases (actual)\n+ actual - scheduled\n(B, 36, 2)", "#e9f1e4")
box(0.1, 2.1, 2.2, 0.9, "Announced future releases\n(B, 6, 3): rate, lead,\nannounced-at / effective-at", "#f3e9d6")
box(0.1, 0.9, 2.2, 0.9, "Static: river graph,\ndistances, positions\n(7 nodes, 6 edges)", "#eeeeee")
# per-node temporal encoder
box(3.0, 4.2, 2.0, 2.5, "Per-node temporal\nencoder (GRU-D style,\nshared weights +\nnode-type embedding)\nmissing -> mask, age,\ndecay", fs=8)
for y in (6.15, 4.95, 3.75):
    arr(2.3, y, 3.0, min(max(y, 4.6), 6.3))
# graph
box(5.6, 3.6, 2.6, 3.1, "Directed river graph\nmessage passing (K=4),\nupstream -> downstream\nonly; edge feats: distance,\nlearned lag attention over\nlast 36 steps; latent node U\n(unmonitored tributary)\n(B, 7 nodes, 64)", fs=7.5)
arr(5.0, 5.4, 5.6, 5.4); ax.text(5.3, 5.55, "(B,7,64)", fontsize=7, ha="center", color="#7a2e2e")
# future release encoder
box(3.0, 1.9, 2.0, 1.2, "Release-schedule encoder\n(B, 6, 3) -> (B, 4, 64)\ndelay-gated per station", fs=7.5)
arr(2.3, 2.55, 3.0, 2.5)
ax.plot([2.3, 6.9], [1.35, 1.35], color="#26496b", lw=1.1); arr(6.9, 1.35, 6.9, 3.6)
# head
box(8.9, 3.6, 2.0, 3.1, "Forecast head\nper station i:\nDelta_i = MLP([z_i, f_i])\nquantiles 0.1/0.5/0.9\ny_i = level_i(t0)+Delta_i\n(B, 4, 3)", "#f9e0d0", fs=8)
arr(8.2, 5.1, 8.9, 5.1); ax.text(8.55, 5.25, "z", fontsize=7, ha="center", color="#7a2e2e")
arr(5.0, 2.5, 8.9, 3.9); ax.text(7.5, 2.9, "f: (B,4,64)", fontsize=7, color="#7a2e2e")
box(11.4, 4.3, 1.5, 1.7, "S1 S2\nS3 S4\nlevel at\nt0 + 60 min", "#f9e0d0", fs=9)
arr(10.9, 5.15, 11.4, 5.15)
ax.text(6.5, 0.4, "t0 = issue time (every 10 min). B = batch of issue times in training, B=1 in operation.", ha="center", fontsize=8)
ax.text(6.5, 7.0, "Figure Q1: forecasting architecture", ha="center", fontsize=11, weight="bold")
plt.savefig("q1_architecture.png", dpi=160, bbox_inches="tight")
