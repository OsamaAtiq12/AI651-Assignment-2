# AI651 Assignment 2 — Osama Atiq (25280091)

Public code for the PDF report in `report/DL4STG_PA2_report.pdf`.

Question 1 is a design, not a training run. Question 2 is Moving MNIST. The numbers in the PDF are the files in `q2/` listed below. Executed notebooks that display the same results are `notebooks/Question1.ipynb` and `notebooks/Question2.ipynb`.

## Setup

From this folder:

```
pip install -r requirements.txt
```

Moving MNIST is downloaded by torchvision into `q2/data`, or into the directory in the environment variable `MOVING_MNIST_ROOT`.

A CUDA GPU was used for the submitted runs (NVIDIA GeForce RTX 5070). The same scripts run on CPU.

## How the submitted results were produced

Run these from `q2/`:

```
python table1_fig1.py
python train.py --model A --seed 0
python train.py --model A --seed 1
python table2_fig2.py
python sec3.py
python choose_d.py
python choose_d2.py
python train.py --model B --seed 0
python train.py --model B --seed 1
python train.py --model C --seed 0
python train.py --model C --seed 1
python train.py --model D --seed 0
python train.py --model D --seed 1
python train.py --model DB --seed 0
python train.py --model DB --seed 1
python sec5.py
```

`train.py` trains one model for 30 epochs (Adam, learning rate 1e-3, batch 32) and writes a log. Models: `A` baseline, `B` symmetry augmentation, `C` speed augmentation, `D` larger reach, `DB` model D plus symmetry augmentation. Seeds are 0 and 1.

`choose_d.py` only searches dilated kernels and writes `d_design_round1.json`. It does not choose the model D in the tables. `choose_d2.py` reads that file, tries the dense kernels, and writes `d_design.json`. The submitted model D is `k35_h48` from that second file. Training D before `choose_d2.py` has finished loads a different design.

## Where the numbers live

| PDF item | File |
|---|---|
| Table 1 and Figure 1 | `q2/table1.json`, `q2/figure1.json` |
| Table 2 | `q2/table2.json` |
| Table 3 | `q2/table3.json` |
| Table 4 | `q2/table4.json` |
| Table 5 | `q2/table5.json` |
| Table 6 | `q2/table6.json` |
| Per-epoch training loss, validation E5, and seconds | `q2/trainA_s0.log`, `q2/trainA_s1.log`, and the same pattern for B, C, D, and DB |
| Same training record as JSON, including GPU name | `q2/ckpt/A_s0.json` and the matching files for every model and seed |
| Model D design search | `q2/d_design.json`, `q2/choose_d.log`, `q2/choose_d2.log` |
| Figure 3 and Figure 6a GIFs | `q2/gifs_A/`, `q2/gifs_final/` |

Weight files (`*.pt`) are not in the repository. `train.py` writes them next to the JSON logs, and `.gitignore` excludes them because of size. The logs and the table JSON files are the record behind every reported number.
