"""Pick model D's design on the VALIDATION split only (never the test split)."""
import json, torch
from common import *
from models import Forecaster, n_params, reach_px
import train as T

cands = {"d22": (2, 2), "d23": (2, 3), "d33": (3, 3)}
va = split(load_moving_mnist())[1].to(T.DEV)
res = {}
for nm, d in cands.items():
    m = Forecaster(dils=d)
    assert n_params(m) <= 721176
    T.SPEED_AUG["D"], T.SYM_AUG["D"] = True, False
    m, _ = T.train("D", 0, epochs=15, out_dir="ckpt_cand", quiet=True, model=m, tag=nm)
    m.eval(); r = {"dils": d, "reach_px": reach_px((3, 3), d)}
    with torch.no_grad():
        for k in (1, 2, 3):
            xi, yi = window(None, 1, k)
            X = va[:, [f - 1 for f in xi]]; Y = va[:, [f - 1 for f in yi]]
            P = torch.cat([m(X[i:i+250]) for i in range(0, 1000, 250)])[:, :2]
            r[f"valE2_{k}x"] = ((P - Y[:, :2]) ** 2).mean().item()
    r["score"] = sum(r[f"valE2_{k}x"] for k in (1, 2, 3))
    res[nm] = r; print(nm, r, flush=True)
best = min(res, key=lambda k: res[k]["score"])
d = res[best]["dils"]
json.dump({"hid": 64, "ks": [3, 3], "dils": list(d), "chosen": best, "candidates": res}, open("d_design_round1.json", "w"), indent=1)
print("chosen round 1", best, "(not the submitted model D; run choose_d2.py next)")
