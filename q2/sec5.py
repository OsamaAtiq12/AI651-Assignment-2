"""Table 5 (A-D, optional D+B), Table 6 (symmetrised B, D, final), Figure 6."""
import sys, json, os
import numpy as np, torch
from evalib import *
import sec3

names = sys.argv[1].split(",")          # e.g. A,B,C,D  or A,B,C,D,DB
final = sys.argv[2] if len(sys.argv) > 2 else None   # "D" or "DB"


def row(f, name, seed, sym=False):
    r = {"E5": E5(f)}
    for T, ck, ek in (("M", "CM", "EM"), ("R", "CR", "ER")):
        c, e = CE(f, sec3.TRANS[T]); r[ck] = c; r[ek] = e
    for k in (1, 2, 3): r[f"E2_{k}x"] = E2(f, k)
    base = f.f if sym else f
    r["params"] = n_params(base)
    r["train_s_per_epoch"] = train_epoch_time(name, seed)
    r["inference_s"] = inference_time(f)
    return r


out = {"table5": {}, "table6": {}}
if not final:
    for n in names:
        for s in (0, 1):
            out["table5"][f"{n}_s{s}"] = row(load_model(n, s), n, s)
else:
    for n in names:
        if n in ("A", "C"): continue
        for s in (0, 1):
            out["table6"][f"sym_{n}_s{s}"] = row(Sym(load_model(n, s)), n, s, sym=True)
    # Figure 6 with the final model, seed 0
    f = Sym(load_model(final, 0))
    picks = [tuple(p) for p in json.load(open("fig3_picks.json"))]
    os.makedirs("gifs_final", exist_ok=True)
    sec3.fig3(f, picks, "figure6a.png", "gifs_final/figure6a", f"Figure 6a (final model = symmetrised {final}, seed 0)")
    n5 = json.load(open("fig5_pick.json"))["fig5_video"]
    sec3.fig5(f, n5, "figure6b.png", f"Figure 6b: final model, test video {n5}")
json.dump(out, open("table5.json" if not final else "table6.json", "w"), indent=1)
for k, v in out["table5"].items(): print(k, {a: round(b, 5) if isinstance(b, float) else b for a, b in v.items()})
for k, v in out["table6"].items(): print(k, {a: (float('%.3g' % b) if isinstance(b, float) else b) for a, b in v.items()})
