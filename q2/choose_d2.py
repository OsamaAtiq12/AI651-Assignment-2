"""Round 2 of D design choice (validation split only): dense larger kernels, narrower hidden width."""
import json, torch
from common import *
from models import Forecaster, n_params, reach_px
import train as T

r1 = json.load(open("d_design_round1.json"))
res = dict(r1["candidates"])
cands = {"k35_h48": dict(hid=48, ks=(3, 5), dils=(1, 1)), "k55_h40": dict(hid=40, ks=(5, 5), dils=(1, 1))}
va = split(load_moving_mnist())[1].to(T.DEV)
T.SPEED_AUG["D"], T.SYM_AUG["D"] = True, False
for nm, c in cands.items():
    m = Forecaster(**c); assert n_params(m) <= 721176, n_params(m)
    m, _ = T.train("D", 0, epochs=15, out_dir="ckpt_cand", quiet=True, model=m, tag=nm)
    m.eval(); r = {**{k: list(v) if isinstance(v, tuple) else v for k, v in c.items()}, "params": n_params(m), "reach_px": reach_px(c["ks"], c["dils"])}
    with torch.no_grad():
        for k in (1, 2, 3):
            xi, yi = window(None, 1, k)
            X = va[:, [f - 1 for f in xi]]; Y = va[:, [f - 1 for f in yi]]
            P = torch.cat([m(X[i:i+250]) for i in range(0, 1000, 250)])[:, :2]
            r[f"valE2_{k}x"] = ((P - Y[:, :2]) ** 2).mean().item()
    r["score"] = sum(r[f"valE2_{k}x"] for k in (1, 2, 3))
    res[nm] = r; print(nm, r, flush=True)
best = min(res, key=lambda k: res[k]["score"]); b = res[best]
cfg = {"hid": b.get("hid", 64), "ks": b.get("ks", [3, 3]), "dils": list(b.get("dils", (1, 1))), "chosen": best, "candidates": res}
json.dump(cfg, open("d_design.json", "w"), indent=1); print("chosen", best)
