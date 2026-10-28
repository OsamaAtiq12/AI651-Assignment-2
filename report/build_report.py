import json, re, os, html
import numpy as np

Q2 = os.path.join(os.path.dirname(__file__), "..", "q2")
J = lambda f: json.load(open(os.path.join(Q2, f)))
t2, t3, t4, t5, t6 = J("table2.json"), J("table3.json"), J("table4.json"), J("table5.json")["table5"], J("table6.json")["table6"]
t1, f1, dd = J("table1.json"), J("figure1.json"), J("d_design.json")
picks3 = J("fig3_picks.json"); v5 = J("fig5_pick.json")["fig5_video"]; f2p = J("fig2_picks.json")["picks"]


def g(x):
    if x == 0: return "0"
    if abs(x) < 1e-3: return f"{x:.2e}".replace("e-0", "e-")
    return f"{x:#.3g}".rstrip(".") if abs(x) < 100 else f"{x:.0f}"


def mean2(a, b): return (a + b) / 2


def md(text):  # tiny markdown -> html (bold, code, line breaks)
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


def paras(text):
    return "".join(f"<p>{md(p.strip())}</p>" for p in text.strip().split("\n\n"))


def table(head, rows, cap, small=False):
    h = "".join(f"<th>{x}</th>" for x in head)
    body = ""
    for r in rows:
        cls = ""
        if isinstance(r, tuple): r, cls = r
        body += f"<tr class='{cls}'>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
    return f"<div class='cap'><b>{cap[0]}.</b> {cap[1]}</div><table class='{'small' if small else ''}'><tr>{h}</tr>{body}</table>"


def fig(path, cap, w="100%"):
    return f"<figure><img src='../q2/{path}' style='width:{w}'><figcaption><b>{cap[0]}.</b> {cap[1]}</figcaption></figure>"


# --------------------------- derived numbers
A = {k: mean2(t5["A_s0"][k], t5["A_s1"][k]) for k in t5["A_s0"]}
M = lambda name: {k: mean2(t5[name + "_s0"][k], t5[name + "_s1"][k]) for k in t5[name + "_s0"]}
B, C, D, DB = M("B"), M("C"), M("D"), M("DB")
S = lambda name: {k: mean2(t6["sym_" + name + "_s0"][k], t6["sym_" + name + "_s1"][k]) for k in t6["sym_" + name + "_s0"]}
sB, sD, sDB = S("B"), S("D"), S("DB")
A2 = np.mean([t2["A_s0"]["mse"], t2["A_s1"]["mse"]], 0)
blank_E2 = float(np.mean(t2["Blank"]["mse"][:2]))
cl = {"E2_1x": t4["E2_1x"][2], "E2_2x": t4["E2_2x"][2], "E2_3x": t4["E2_3x"][2], "E5_1x": t4["E5_1x"][2], "E5_2x": t4["E5_2x"][2]}
a_tab4 = {k: mean2(t4[k][0], t4[k][1]) for k in t4}
pct = lambda new, old: f"{100*(new-old)/old:+.0f}%"

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.2pt; line-height: 1.38; color:#111; }
h1 { font-size: 17pt; margin: 0 0 4pt; } h2 { font-size: 13.5pt; border-bottom: 1.5px solid #26496b; color:#26496b; margin-top: 16pt; page-break-after: avoid;}
h3 { font-size: 11.2pt; color:#26496b; margin: 11pt 0 3pt; page-break-after: avoid;}
code { font-family: Consolas, monospace; font-size: 9pt; background:#f1f1f1; padding: 0 2px;}
table { border-collapse: collapse; margin: 3pt 0 9pt; font-size: 8.6pt; width: 100%; page-break-inside: avoid;}
table.small { font-size: 8pt; }
th, td { border: 0.6px solid #999; padding: 2.5px 4px; text-align: center; } th { background: #e6edf4; }
td:first-child { text-align: left; } tr.sub td { color:#555; font-size: 0.92em; } tr.bold td { font-weight: 600; background:#f3f7fb; }
.cap { font-size: 9pt; margin-top: 6pt; page-break-after: avoid; } figure { margin: 6pt 0; page-break-inside: avoid; text-align: center;}
figcaption { font-size: 8.8pt; text-align: left; margin-top: 2pt;}
.q { background:#f6f8fb; border-left: 3px solid #26496b; padding: 4pt 8pt; margin: 8pt 0 3pt; font-size: 9.2pt; color:#26496b;}
.verb { border: 1px dashed #888; background:#fffdf3; padding: 5pt 9pt; font-size: 9.4pt;}
.todo { background:#fff1c9; padding: 0 3px; }
.pb { page-break-before: always; }
"""

# --------------------------- Tables 1-6
T1 = table(["", "Check", "Expected", "Yours"], [
    ["(a)", "max|M(M(x)) − x|", "0", f"{g(t1['a'])} (torch.equal: {t1['a_equal']})"],
    ["(b)", "max|R⁴(x) − x|", "0", f"{g(t1['b'])} (torch.equal: {t1['b_equal']})"],
    ["(c)", "max|MRM(x) − R⁻¹(x)|", "0", f"{g(t1['c'])} (torch.equal: {t1['c_equal']})"],
    ["(d)", "bright pixel of M(p) (frame, row, col)", "(7, 10, 43)", str(tuple(t1['d']))],
    ["(e)", "bright pixel of R(p) (frame, row, col)", "(7, 43, 10)", str(tuple(t1['e']))],
    ["(f)", "1× input / target frames", "1–5 / 6–10", f"{t1['f'][0]} / {t1['f'][1]}"],
    ["(g)", "2× input / target frames", "1,3,5,7,9 / 11,13,15,17,19", f"{t1['g'][0]} / {t1['g'][1]}"],
    ["(h)", "3× input / target frames", "1,4,7,10,13 / 16,19", f"{t1['h'][0]} / {t1['h'][1]}"],
], ("Table 1", "Sanity checks of the transformation and window functions (the same functions are used in every experiment)."))

hdr2 = ["Predictor", "MSE₁", "MSE₂", "MSE₃", "MSE₄", "MSE₅", "E⁽⁵⁾", "Params"]
row2 = lambda nm, d, p, cls="": ([nm] + [g(x) for x in d["mse"]] + [g(d["E5"]), p], cls)
T2 = table(hdr2, [
    row2("Model A, mean(seed 0, seed 1)", {"mse": A2.tolist(), "E5": float(A2.mean())}, "480,784", "bold"),
    row2("  seed 0", t2["A_s0"], "480,784", "sub"), row2("  seed 1", t2["A_s1"], "480,784", "sub"),
    row2("Blank", t2["Blank"], "0"), row2("Copy-last", t2["Copy-last"], "0")],
    ("Table 2", "Per-horizon test MSE, E⁽⁵⁾ and trainable parameters (1,000 test videos, 1×, t = 1)."))

hdr3 = ["Predictor", "E⁽⁵⁾", "C_M", "E_M", "C_R", "E_R"]
def row3(nm, d, cls=""): return ([nm] + [g(d[k]) for k in ("E5", "CM", "EM", "CR", "ER")], cls)
a3 = {k: float(np.mean(t3["A"][k])) for k in t3["A"]}
T3 = table(hdr3, [
    row3("Model A, mean(seed 0, seed 1)", a3, "bold"),
    row3("  seed 0", {k: t3["A"][k][0] for k in t3["A"]}, "sub"), row3("  seed 1", {k: t3["A"][k][1] for k in t3["A"]}, "sub"),
    row3("Blank", {k: t3["Blank"][k][0] for k in t3["Blank"]}), row3("Copy-last", {k: t3["Copy-last"][k][0] for k in t3["Copy-last"]})],
    ("Table 3", "Accuracy E⁽⁵⁾, consistency gaps C_M, C_R and transformed errors E_M, E_R (1×, t = 1, test set)."))

hdr4 = ["Predictor", "E⁽²⁾ 1×", "E⁽²⁾ 2×", "E⁽²⁾ 3×", "E⁽⁵⁾ 1×", "E⁽⁵⁾ 2×"]
row4 = lambda nm, d, cls="": ([nm] + [g(d[k]) for k in ("E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x")], cls)
T4 = table(hdr4, [
    row4("Model A, mean(seed 0, seed 1)", a_tab4, "bold"),
    row4("  seed 0", {k: t4[k][0] for k in t4}, "sub"), row4("  seed 1", {k: t4[k][1] for k in t4}, "sub"),
    row4("Copy-last", cl)],
    ("Table 4", "Model A at faster motion (t = 1; at 3× only the first two horizons exist, so E⁽⁵⁾ is reported at 1× and 2× only)."))

names = [("A", A, "A"), ("B", B, "B"), ("C", C, "C"), ("D", D, "D"), ("D+B", DB, "DB")]
def rows5(a_or_b):
    rows = []
    for lab, m, key in names:
        s0, s1 = t5[key + "_s0"], t5[key + "_s1"]
        cols = ("E5", "CM", "EM", "CR", "ER") if a_or_b == "a" else ("E2_1x", "E2_2x", "E2_3x", "params", "train_s_per_epoch", "inference_s")
        fm = (lambda k, d: f"{d[k]:,.0f}" if k == "params" else g(d[k]))
        rows.append(([f"{lab}, mean"] + [fm(k, m) for k in cols], "bold"))
        rows.append(([f"&nbsp;&nbsp;seed 0"] + [fm(k, s0) for k in cols], "sub"))
        rows.append(([f"&nbsp;&nbsp;seed 1"] + [fm(k, s1) for k in cols], "sub"))
    return rows
T5a = table(["Model", "E⁽⁵⁾ (1×)", "C_M", "E_M", "C_R", "E_R"], rows5("a"), ("Table 5 (accuracy and consistency)", "Models A–D and the optional D+B, each with seeds 0 and 1 and their mean; test set, 1×, t = 1."))
T5b = table(["Model", "E⁽²⁾ 1×", "E⁽²⁾ 2×", "E⁽²⁾ 3×", "Params", "Train s/epoch", "Inference s"], rows5("b"), ("Table 5 (speed and cost)", "Same models. Hardware: NVIDIA RTX 5070 (12 GB), PyTorch 2.11 + CUDA 12.8, Windows 11. Train s/epoch = mean wall-clock over 30 epochs; inference = time to predict the 1,000 test videos at batch size 32 under model.eval() and torch.no_grad() (fastest of 5 timed passes after a warm-up)."))
def rows6():
    rows = []
    for lab, m, key in (("Symmetrised B", sB, "B"), ("Symmetrised D", sD, "D"), ("Final: symmetrised D+B", sDB, "DB")):
        s0, s1 = t6["sym_" + key + "_s0"], t6["sym_" + key + "_s1"]
        cols = ("E5", "CM", "EM", "CR", "ER", "E2_1x", "E2_2x", "E2_3x", "params", "train_s_per_epoch", "inference_s")
        fm = (lambda k, d: f"{d[k]:,.0f}" if k == "params" else g(d[k]))
        rows.append(([lab + ", mean"] + [fm(k, m) for k in cols], "bold"))
        rows.append(([f"&nbsp;&nbsp;seed 0"] + [fm(k, s0) for k in cols], "sub"))
        rows.append(([f"&nbsp;&nbsp;seed 1"] + [fm(k, s1) for k in cols], "sub"))
    return rows
T6 = table(["Model", "E⁽⁵⁾", "C_M", "E_M", "C_R", "E_R", "E⁽²⁾ 1×", "E⁽²⁾ 2×", "E⁽²⁾ 3×", "Params", "Train s/ep", "Infer s"], rows6(),
           ("Table 6", "Symmetrised forecasters (eq. 7). Training time is that of the underlying learned model; inference includes all eight transformations, forward passes, inverse transformations and the average. C_M and C_R are zero up to floating-point rounding."), small=True)

dcand = dd["candidates"]
def drow(nm, c):
    return [nm, c.get("hid", 64), str(c.get("ks", [3, 3])), str(c.get("dils", [1, 1])), f"{c.get('reach_px')}", g(c["valE2_1x"]), g(c["valE2_2x"]), g(c["valE2_3x"]), g(c["score"])]
DT = table(["Candidate", "Hidden", "Kernels", "Dilations", "Reach (px)", "val E⁽²⁾ 1×", "2×", "3×", "Sum"],
           [drow(k, v) for k, v in dcand.items()], ("Table D", "Model D design search (VALIDATION split only, seed 0, 15 epochs, speed augmentation). Chosen: " + dd["chosen"] + " (lowest sum)."), small=True)

pred5 = open(os.path.join(Q2, "..", "predictions", "Q2.5_prediction.md"), encoding="utf8").read()
pred11 = open(os.path.join(Q2, "..", "predictions", "Q2.11_prediction.md"), encoding="utf8").read()
verb = lambda s: "<div class='verb'>" + paras("\n\n".join(l for l in s.split("\n\n") if not l.startswith("#"))) + "</div>"
Q = lambda s: f"<div class='q'>{s}</div>"

fr = f1["dir_fractions"]
html_doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>AI651 Assignment 2</title><style>{CSS}</style></head><body>
<h1>AI651 – Deep Learning for Space, Time and Graphs · Assignment 2</h1>
<p><b>Student:</b> Osama Atiq (25280091)<br>
<b>Code repository (public):</b> <a href="https://github.com/OsamaAtiq12/AI651-Assignment-2">https://github.com/OsamaAtiq12/AI651-Assignment-2</a><br>
All code, logs, GIFs and the committed predictions (Q2.5, Q2.11) are in the repository: <code>q2/</code> (code, results), <code>predictions/</code>, <code>report/</code>.</p>

<h2>Question 1 — River Forecasting Proposal</h2>
@@Q1@@

<h2 class='pb'>Question 2 — Motion or Memory?</h2>
<p>Setup: Moving MNIST (10,000 videos); videos 0–7,999 train, 8,000–8,999 validation, 9,000–9,999 test; frames scaled to [0,1]. Every learned model uses the handout's protocol (Adam 1e-3, batch 32, 30 epochs, seeds 0 and 1, free-running 9-step rollout, masked loss). Hardware: RTX 5070 12 GB, PyTorch 2.11 (CUDA 12.8). Tables show mean(seed 0, seed 1) in bold with each seed beneath; figures, GIFs and per-video analyses use the seed-0 model. The test split was used only for evaluation; model D's design was chosen on validation (Table D).</p>

<h3>1. Setup</h3>
{T1}
<p>All eight checks match the expected values (maximum absolute differences are exactly 0 and <code>torch.equal</code> returns True).</p>
{fig("figure1.png", ("Figure 1", f"Optical flow (Farneback, handout settings) pooled over the first 50 training videos (950 frame pairs). (a) flow direction θ = atan2(−d_y, d_x), 18 bins of 20°, fraction of kept pixels, red dashed line = uniform 1/18; (b) flow speed in pixels per frame. Median speed = {g(f1['median_speed'])} px/frame; number of kept pixels = {f1['n_kept']:,}."))}
{Q("Q2.1")}
<p><b>Mirrored and rotated trajectories already look familiar.</b> A mirror maps a flow direction θ to 180° − θ, and a 90° rotation maps θ to θ + 90°. If every training video were mirrored or rotated, the histogram would be concentrated around particular angles; in Figure 1(a) every one of the 18 bins contains substantial mass: the smallest bin holds {g(min(fr))} of the pixels (120–140°, about half the uniform level 0.0556) and the largest {g(max(fr))} (180–200°). The histogram is not perfectly flat (only 50 videos), but no direction is missing, so the mirrored and rotated versions of the training motion are directions the model has already seen (its motion is, at least approximately, direction-isotropic). <b>Typical speed</b> is the median {g(f1['median_speed'])} px/frame (5–95 % range {g(f1['speed_pcts'][0])}–{g(f1['speed_pcts'][4])} px/frame; Figure 1(b) shows almost no flow above 5 px/frame). Hence 2× corresponds to about 2 × {g(f1['median_speed'])} ≈ {g(2*f1['median_speed'])} px/frame and 3× to ≈ {g(3*f1['median_speed'])} px/frame, i.e. beyond everything in the training data at 1×. <b>Overlaps:</b> optical flow is unreliable where the digits overlap (they pass through each other, so intensities blend and the flow can be wrong). That affects a small fraction of the kept pixels, and it could blur the histogram or shift a few pixels in magnitude; it cannot fill the empty direction range or move the median from ≈3.3 to ≈6.6 px/frame, so the conclusions (all directions present, speeds ≈ 3–4 px/frame) would not change. A caveat in the other direction: Farneback with a 15-pixel window tends to underestimate large displacements, which would, if anything, make the true speeds somewhat higher than shown.</p>
{Q("Q2.2")}
<p><b>Mirror</b> changes the <i>appearance</i> (a mirrored 2 is not a 2) but not the motion statistics: speeds are unchanged and directions are mapped to 180° − θ, which the training data already contains (Q2.1). <b>Rotation</b> changes the appearance even more (strokes become sideways) and, again, leaves the speed distribution unchanged and rotates the directions by 90°, which is also covered by the training data. <b>Speed-up</b> changes the <i>motion</i> only: the digits look exactly as before, but they move k times further per frame, to speeds (≈6.6 and ≈10 px/frame) not in the training data. This makes the three transformations a controlled experiment: mirror/rotation vary appearance with (almost) fixed motion, speed-up varies motion with fixed appearance, so a failure under one but not the other tells us which factor the model depends on.</p>

<h3>2. Baseline: ConvLSTM</h3>
{T2}
<p>Model A has exactly 480,784 trainable parameters (184,576 in layer 1, 295,168 in layer 2, 1,040 in the 1×1 convolution), matching the handout.</p>
{fig("figure2.png", ("Figure 2", f"Seed-0 model A on three test videos (test indices {', '.join(str(9000+i) for i in f2p)}): row 1 = input frames x₁–x₅ (columns 1–5), row 2 = predictions x̂₆–x̂₁₀ (columns 6–10), row 3 = ground truth x₆–x₁₀. Video #120 (dataset index 9120) has overlapping digits already in the input."))}
{Q("Q2.3")}
<p><b>Reach of model A = 8 original pixels per time step.</b> Work in the folded 16×16 grid, where one cell is 4 original pixels (centre-to-centre; the 4×4 patch width is not added). Only the 3×3 convolutions move information (radius 1 cell each); the 1×1 output convolution, <code>pixel_unshuffle</code> and <code>pixel_shuffle</code> do not. Paths within one step τ: (i) from the new input u_τ: layer-1 input convolution (1 cell) then layer-2 input convolution on h⁽¹⁾_τ (1 cell) = 2 cells; (ii) from the previous state h⁽¹⁾_τ₋₁: layer-1 recurrent convolution (1 cell), then layer-2's input convolution (1 cell) = 2 cells; (iii) from h⁽²⁾_τ₋₁: layer-2 recurrent convolution only = 1 cell. The longest path is 2 cells = 2 × 4 = <b>8 px</b>. (Over several steps information can travel further, but a digit moving more than 8 px between consecutive frames cannot be connected in a single update.) Compared with the typical speed of 3.3 px/frame (Figure 1) the reach is comfortable at 1×, marginal at 2× (≈6.6 px), and too small at 3× (≈10 px).</p>
{Q("Q2.4")}
<p>Model A (E⁽⁵⁾ = {g(A2.mean())}, mean of seeds) is well below both references: {g(100*(1-A2.mean()/t2['Blank']['E5']))} % lower than Blank ({g(t2['Blank']['E5'])}) and {g(100*(1-A2.mean()/t2['Copy-last']['E5']))} % lower than Copy-last ({g(t2['Copy-last']['E5'])}). Its error grows with the horizon: MSE₁ = {g(A2[0])} to MSE₅ = {g(A2[4])} ({pct(A2[4], A2[0])}); Copy-last grows faster ({g(t2['Copy-last']['mse'][0])} to {g(t2['Copy-last']['mse'][4])}, {pct(t2['Copy-last']['mse'][4], t2['Copy-last']['mse'][0])}) while Blank is flat at ≈0.0426. A stays below Blank at all five horizons, but the gap narrows (MSE₅ is {g(100*A2[4]/t2['Blank']['mse'][4])} % of Blank's). In Figure 2 the predictions are blurry blobs whose position is roughly right but whose shape is lost, and they get fuzzier with the horizon. When digits overlap (video #120) the prediction is a single vertical smear: the model does not keep two digits apart, so it cannot let them pass through each other and emerge on the other side.</p>
{Q("Q2.5 (written before running any experiment in Section 3 — unedited)")}
{verb(pred5)}
<p style='font-size:9pt'>Committed before the Section 3 experiments were run (git commit <code>7752ac2</code>, 2026-10-26 14:37). <i>Outcome:</i> mirror and rotation were as predicted (rotation gap larger than mirror gap, both small); the speed prediction was too optimistic: 2× already degrades to Blank level (Table 4).</p>

<h3>3. Testing the Baseline</h3>
{T3}
{fig("figure3.png", ("Figure 3", "Seed-0 model A. The two test videos with the largest per-video C_M (T = M; test indices 673 and 980, i.e. dataset indices 9673 and 9980) and the two with the largest C_R (T = R; test indices 254 and 982). Per video: row 1 X | f(X); row 2 T(X) | T(Y); row 3 T(X) | f(T(X)); row 4 T(X) | T(f(X)); columns 1–5 inputs, 6–10 forecasts. GIFs stepping through the five horizons: q2/gifs_A/figure3_video673_M.gif, figure3_video980_M.gif, figure3_video254_R.gif, figure3_video982_R.gif."))}
{fig("figure4.png", ("Figure 4", "Histograms of per-video C_M (left) and C_R (right) of the seed-0 model A over the 1,000 test videos; dashed lines mark the Figure 3 videos (solid-red = the transformation the video was chosen for, orange dotted = the other one)."), "80%")}
{Q("Q2.6")}
<p>Blank and Copy-last are exactly consistent (C_M = C_R = 0) because they act pixel by pixel and in a symmetric way, so transforming first or last gives identical outputs — yet they are poor predictors (E⁽⁵⁾ = {g(t3['Blank']['E5'][0])} and {g(t3['Copy-last']['E5'][0])}). So a small C_T on its own says only that the model treats a scene and its transformed copy alike; it does not say that either prediction is right. A model can be perfectly consistent and wrong. E_T = MSE(f(T(X)), T(Y)) compares the prediction on the transformed scene with the correct answer, so it must always be reported next to C_T. For model A, C_M = {g(a3['CM'])} and C_R = {g(a3['CR'])} are small next to E⁽⁵⁾ = {g(a3['E5'])}, while E_M = {g(a3['EM'])} and E_R = {g(a3['ER'])} are within {g(100*(a3['ER']/a3['E5']-1))} % of E⁽⁵⁾.</p>
{Q("Q2.7")}
<p><b>Video 980 (dataset index 9980), T = M (Figure 3, second block).</b> The true future (row 2) shows the two overlapping digits separating: one piece moves up and to the right while the other continues lower down, so two recognisable shapes appear. Row 4 (the mirror of the prediction on the original input) is a narrow vertical streak that stays thin with a short tail at the top: the model's account is that the digits stay merged in one place and fade. Row 3 (prediction on the mirrored input) is a blob that starts compact and then bends into a curved, wider crescent drifting to one side. So, with the same motion seen in a mirrored scene, the model's account changes from <i>a digit-pair that blurs into one narrow smear in place</i> to <i>a blob that curves and spreads</i>; neither shows the two digits parting, and in both the digit shapes blur into something unrecognisable.</p>
{Q("Q2.8")}
<p>Rotation is harder than mirroring, but only modestly: for model A, C_R = {g(a3['CR'])} against C_M = {g(a3['CM'])} ({g(a3['CR']/a3['CM'])}×), and E_R = {g(a3['ER'])} against E_M = {g(a3['EM'])}; the same ordering holds for each seed (Table 3), and Figure 4 shows a heavier tail for C_R. By Q2.1 and Q2.2 the <i>motion</i> statistics are about equally familiar under both transformations (all directions occur, speeds unchanged), so I attribute most of the difference to <i>appearance</i>: a rotated digit is further from anything in the training set than a mirrored one (mirror keeps the stroke orientation statistics, rotation turns vertical strokes into horizontal ones), and model A's blob-like predictions depend on those shapes. Motion cannot be fully excluded, because Figure 1(a) is not perfectly flat (some of the 50-video sample's directions are under-represented), but model B supports the appearance reading: symmetry augmentation, which changes only appearance and direction labels, lowers C_R from {g(A['CR'])} to {g(B['CR'])} ({pct(B['CR'], A['CR'])}) and C_M from {g(A['CM'])} to {g(B['CM'])} ({pct(B['CM'], A['CM'])}) (Table 5).</p>

<p><b>Faster motion.</b></p>
{T4}
{fig("figure5.png", ("Figure 5", f"Seed-0 model A on test video {v5} (dataset index {9000+v5}), whose digits do not overlap in any displayed frame (checked by connected components after a 7×7 dilation). For each speed: five input frames, then predicted and true first forecast horizon, and predicted and true second forecast horizon."))}
{Q("Q2.9")}
<p><b>Model A falls apart at 2×, earlier than the reach estimate suggested.</b> E⁽²⁾ is {g(a_tab4['E2_1x'])} at 1×, {g(a_tab4['E2_2x'])} at 2× and {g(a_tab4['E2_3x'])} at 3× (Table 4); at 2× it is already no better than Blank (E⁽²⁾ = {g(blank_E2)}) and at 3× it is clearly worse than Blank, though still better than Copy-last ({g(cl['E2_3x'])}). The reach of 8 px (Q2.3) suggested that 2× (≈6.6 px/frame) would be marginal and 3× (≈10 px) would fail; the measurement is worse than that. What the estimate missed is that 8 px is an <i>upper bound</i> reached only along the longest path (input → layer 1 → layer 2); the recurrent route that carries the digit's old position (h⁽²⁾ → h⁽²⁾) reaches only 4 px, and the ends of the 3×3 kernels see a digit's stroke only partially, so reliable tracking works for displacements clearly below 8 px, not at 6–7 px. In Figure 5 (video {v5}): at 1× the two digits are in about the right place but blurred; at 2× the prediction is a smeared vertical streak that spans the gap between where a digit was and where it goes, i.e. a <i>duplicated / blurred digit</i> between old and new positions rather than a digit at the new position; at 3× the blobs sit near the digits' last input positions and the true digits (which have moved much further, and by h2 are close together) are missing, i.e. the digits <i>stall</i> at the old positions. This suggests that when the new position falls outside what the network can connect to the old one, it cannot follow the digit, and hedges (averaging possible positions → blur) or copies its last state (stalling).</p>
{Q("Q2.10")}
<p>No. Both explanations predict the same symptom (A fails at 2× and 3×): fast motion is missing from the training data <i>and</i> exceeds A's reach, so the speed failure of A alone cannot separate them. Experiment: hold one factor fixed and vary the other. (1) <i>Data fixed, architecture varied:</i> train the larger-reach network of model D on 1× data only (no speed augmentation) and test it at 2× and 3×; if it keeps up where A does not, reach was the limit. (2) <i>Architecture fixed, data varied:</i> train A with speed augmentation (this is model C) and test at 2× and 3×; if it improves a lot, unfamiliar motion was the problem. A further check: speeds seen at neither training nor reach (e.g. 4×) and a non-integer speed such as 1.5× (made by interpolation, so the same frames are not skipped) to show how the models extrapolate or interpolate.</p>

<h3>4. What Can Data Fix?</h3>
<p><b>Model D design.</b> Candidate designs (all keeping A's input/output shapes, 9-step free-running rollout and training protocol; params ≤ 721,176; no attention) were trained with speed augmentation and compared on the <b>validation split only</b>. The dilated candidates are a cautionary tale: with dilation 2/3 in the second layer (d23, d33) the training stalled at the Blank level (validation E⁽²⁾ ≈ 0.0375) and d22 was clearly worse on 1× than A; the dense larger kernels worked better.</p>
{DT}
{Q("Q2.11 (written before training any of B, C, D — unedited)")}
{verb(pred11)}
<p style='font-size:9pt'>Committed before B–D were trained (git commit <code>fb01da2</code>, 2026-10-26 14:38). <i>Outcome:</i> B did not fix speed and reduced the consistency gaps (as predicted); C fixed most of the 2× and 3× failure (I predicted it would stay poor at 3×); D was only slightly better than C at 3× (I predicted a clear win).</p>
{T5a}{T5b}
{Q("Q2.12")}
<p><b>Model D</b> is model A with the same shapes, folding (pixel-unshuffle 4), 9-step free-running rollout and training protocol, but with a wider kernel in the second ConvLSTM layer and a narrower hidden width to stay within the parameter budget: layer 1 = ConvLSTM, 16 → 48 channels, 3×3 kernel; layer 2 = ConvLSTM, 48 → 48 channels, <b>5×5</b> kernel; 1×1 convolution 48 → 16, pixel-shuffle(4), sigmoid. It has <b>{D['params']:,.0f}</b> trainable parameters (≤ 721,176). <b>Reach (same convention as Q2.3):</b> layer-1 radius 1 cell, layer-2 radius 2 cells. From u_τ: layer 1 (1) + layer 2 (2) = 3 cells; from h⁽¹⁾_τ₋₁: layer-1 recurrent (1) + layer-2 input (2) = 3 cells; from h⁽²⁾_τ₋₁: layer-2 recurrent (2) = 2 cells. The longest path is 3 cells = 3 × 4 = <b>12 px</b>, strictly larger than model A's 8 px. (Even model D's recurrent path, 8 px, matches the whole of A.)</p>
{Q("Q2.13")}
<p><b>Appearance (mirror / rotation failure):</b> model B, which changes only appearance, lowers C_M by {pct(B['CM'], A['CM'])} and C_R by {pct(B['CR'], A['CR'])} (Table 5) with no change in speed behaviour (E⁽²⁾ at 3×: {g(B['E2_3x'])} vs {g(A['E2_3x'])}); symmetrisation (Table 6) removes it completely. <b>Unfamiliar motion (speed failure at 2× and 3×):</b> model C, same architecture as A but trained on speeds 1–3×, lowers E⁽²⁾ from {g(A['E2_2x'])} to {g(C['E2_2x'])} at 2× ({pct(C['E2_2x'], A['E2_2x'])}) and from {g(A['E2_3x'])} to {g(C['E2_3x'])} at 3× ({pct(C['E2_3x'], A['E2_3x'])}); at 3× the model goes from worse than Blank ({g(blank_E2)}) to better. <b>Architectural reach:</b> comparing D with C (same data), E⁽²⁾ improves only from {g(C['E2_3x'])} to {g(D['E2_3x'])} at 3× ({pct(D['E2_3x'], C['E2_3x'])}) and from {g(C['E2_2x'])} to {g(D['E2_2x'])} at 2×: a small effect. So in these results the speed failure of A was mainly unfamiliar motion, with a minor reach component. <i>If C still did badly at 3× while D did well</i> (not what we found), that would show the failure is architectural: C and D see exactly the same data, so the only difference is which input pixels can influence an output pixel in one step. More training data cannot extend that: training changes the function computed from the inputs that are connected, but not the set of connected inputs; if the new position of a digit lies outside the one-step reach of its old position, no weights can link them in one update. A caveat on the whole picture: even C and D remain far worse at 3× (E⁽²⁾ ≈ 0.032–0.034) than at 1× (≈ 0.023–0.024), so some of the 3× difficulty is not explained by either factor (scored frames are further apart in time at 3×, so the prediction itself is harder).</p>
{Q("Q2.14")}
<p>Yes, speed augmentation costs accuracy on the original 1× clips: E⁽⁵⁾ at 1× rises from {g(A['E5'])} (A) to {g(C['E5'])} (C, {pct(C['E5'], A['E5'])}), {g(D['E5'])} (D, {pct(D['E5'], A['E5'])}) and {g(DB['E5'])} (D+B, {pct(DB['E5'], A['E5'])}); E⁽²⁾ at 1× rises from {g(A['E2_1x'])} to {g(C['E2_1x'])} / {g(D['E2_1x'])} / {g(DB['E2_1x'])}. Symmetry augmentation (B) costs nothing ({g(B['E5'])}). The loss is expected: only about a third of the training windows are 1×, and the same capacity must cover three speeds. It was worth it: averaged over the three speeds, E⁽²⁾ is {g(np.mean([A['E2_1x'], A['E2_2x'], A['E2_3x']]))} for A against {g(np.mean([DB['E2_1x'], DB['E2_2x'], DB['E2_3x']]))} for D+B, a gain of {g(abs(100*(np.mean([DB['E2_1x'],DB['E2_2x'],DB['E2_3x']])/np.mean([A['E2_1x'],A['E2_2x'],A['E2_3x']])-1)))} %, at a 1× cost of about 0.005. Whether it is worth it in practice depends on whether faster motion occurs at test time; if the application is purely 1×, model A (or B) is the better choice.</p>

<h3>5. The Final Model</h3>
<p><b>Proof that f̄ is exactly consistent.</b> G is a group (the 8 symmetries of the square), so for h ∈ G: f̄(hX) = (1/8) Σ_g g⁻¹(f(g h X)). Substituting g′ = g h (a bijection of G, so g = g′h⁻¹ and g⁻¹ = h g′⁻¹) gives f̄(hX) = (1/8) Σ_g′ h g′⁻¹(f(g′X)) = h · f̄(X), because h is a linear pixel permutation and so commutes with the average. Taking h = M or R gives f̄(T(X)) = T(f̄(X)), i.e. C_M = C_R = 0. In the implementation the same eight terms are summed in a different order, so the result differs only by floating-point rounding (≈10⁻¹⁷ in Table 6).</p>
<p><b>Choice of final model.</b> I trained the optional D+B. Table 5 supports it: its E⁽⁵⁾ ({g(DB['E5'])}) and E⁽²⁾ at all speeds are practically equal to D's ({g(D['E5'])}), its C_R is less than half of D's ({g(DB['CR'])} vs {g(D['CR'])}) and its C_M is about the same ({g(DB['CM'])} vs {g(D['CM'])}), so symmetrisation starts from a model that is at least as close to symmetric; the gain is small, which is why the choice is not critical. The final model is therefore <b>symmetrised D+B</b>.</p>
{T6}
{fig("figure6a.png", ("Figure 6a", "Figure 3 repeated with the final model (symmetrised D+B, seed 0) on exactly the same four videos (same row layout). GIFs: q2/gifs_final/figure6a_video673_M.gif, figure6a_video980_M.gif, figure6a_video254_R.gif, figure6a_video982_R.gif."))}
{fig("figure6b.png", ("Figure 6b", f"Figure 5 repeated with the final model on the same video (test video {v5})."))}
{Q("Q2.15")}
<p><b>Learned symmetry (B)</b> reduces the gaps (C_M {g(A['CM'])} → {g(B['CM'])}, C_R {g(A['CR'])} → {g(B['CR'])}) but they remain non-zero; it costs no inference time but about {g(100*(B['train_s_per_epoch']/A['train_s_per_epoch']-1))} % more training time per epoch ({g(B['train_s_per_epoch'])} vs {g(A['train_s_per_epoch'])} s), and its accuracy on original clips is unchanged (E⁽⁵⁾ {g(B['E5'])} vs {g(A['E5'])}). <b>Built-in symmetry</b> (symmetrised B) gives C_M = C_R = 0 (≈10⁻¹⁷) <i>exactly</i>, by the proof above (it is a property of the wrapper and holds for any f), so E_M = E_R = E⁽⁵⁾; accuracy on original clips is slightly better, E⁽⁵⁾ = {g(sB['E5'])} vs {g(B['E5'])} (averaging eight predictions reduces variance). The price is inference: {g(sB['inference_s'])} s vs {g(B['inference_s'])} s for the 1,000 test videos (≈{g(sB['inference_s']/B['inference_s'])}×). For D+B the gain is {g(sDB['E5'])} vs {g(DB['E5'])}. <b>When averaging can hurt:</b> when the true dynamics are not symmetric (e.g. gravity, a preferred direction, or text with a reading order), forcing equal weights on the eight orientations blends in wrong-orientation predictions and blurs sharp structure. Here the Moving-MNIST dynamics are symmetric and accuracy did not get worse (it improved slightly), although the final model's predictions in Figure 6a/6b are visibly blurrier than model A's in Figure 3/5, which is consistent with the lower sharpness that averaging and speed augmentation produce.</p>
{Q("Q2.16")}
<p><b>No, not fully.</b> In Figure 6a (video 980, T = M) rows 3 and 4 are now identical: the disagreement between f(T(X)) and T(f(X)) seen for model A is gone, and that is what symmetrisation fixes (exact consistency, C_M = 0). What is not fixed is the account of the motion itself: both rows are a featureless grey vertical blob that drifts and spreads slightly, with no sign of the two digits separating and no recognisable digit shapes; the true future (row 2) still shows two distinct shapes parting. The speed-related changes (speed augmentation and larger reach) are not visible on this 1× clip, but they appear in Figure 6b (final model vs Figure 5): at 2× and 3× the final model's predictions are still very blurred (diffuse mass around the region where the digits are), and the lower E⁽²⁾ at 2×/3× ({g(DB['E2_2x'])}, {g(DB['E2_3x'])} vs {g(A['E2_2x'])}, {g(A['E2_3x'])} for A) reflects mostly a better-placed blur rather than sharp digits. Overall: the <i>consistency</i> failure was fixed by the symmetrised wrapper (with B-augmentation helping), the <i>speed</i> failure was mostly fixed by speed augmentation (D's larger reach adding little), and the <i>blur / loss of digit identity</i> was not fixed, and 1× accuracy got worse.</p>
{Q("Q2.17 Final verdict")}
<p>Partly. The final model beats Blank and Copy-last at every speed (E⁽²⁾ at 3× = {g(DB['E2_3x'])} vs {g(blank_E2)} and {g(cl['E2_3x'])}), is exactly mirror/rotation-consistent, and tolerates 2× and 3× playback far better than A: more than memorising 1× clips, though speed augmentation puts those speeds in training. But its predictions are blurry, shapeless blobs, 1× accuracy is worse than model A's, and the digits never separate cleanly after overlap. A sceptic would want unseen speeds (1.5×, 4×), unseen shapes (letters, other digits sets) and time-reversed clips at similar error, with sharp rather than hedged predictions. The unpassed test: speed 4× (≈13 px/frame), beyond both the training speeds and the reach of 12 px.</p>

<h2>References</h2>
<p style='font-size:9pt'>Shi et al. (2015), Convolutional LSTM Network, NeurIPS. Srivastava, Mansimov, Salakhutdinov (2015), Unsupervised Learning of Video Representations using LSTMs, ICML.</p>

<h2>AI-use disclosure (required by the course AI policy)</h2>
<p>I used AI to understand the problem statement, for code completion, to check syntax, and for a small amount of research.</p>
</body></html>"""

q1 = open(os.path.join(os.path.dirname(__file__), "Q1.md"), encoding="utf8").read()
# convert Q1.md (headings, tables, bullets, bold, code) to HTML
def q1_html(t):
    lines, out, i = t.split("\n"), [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("# "): i += 1; continue
        if l.startswith("## "): out.append(f"<h3>{md(l[3:])}</h3>")
        elif l.startswith("!["): out.append("<figure><img src='q1_architecture.png' style='width:100%'><figcaption><b>Figure Q1.</b> Architecture (data sources, processing blocks, source combination, forecasts for S1–S4; main tensor shapes labelled).</figcaption></figure>")
        elif l.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-| ]+\|$", lines[i]): rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            i -= 1
            out.append("<table class='small'><tr>" + "".join(f"<th>{md(c)}</th>" for c in rows[0]) + "</tr>" + "".join("<tr>" + "".join(f"<td style='text-align:left'>{md(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>")
        elif re.match(r"^\d+\. ", l): out.append(f"<p style='margin:2pt 0 2pt 12pt'>{md(l)}</p>")
        elif l.strip(): out.append(f"<p>{md(l)}</p>")
        i += 1
    return "\n".join(out)

html_doc = html_doc.replace("@@Q1@@", q1_html(q1))
open(os.path.join(os.path.dirname(__file__), "report.html"), "w", encoding="utf8").write(html_doc)
print("ok", len(html_doc))
