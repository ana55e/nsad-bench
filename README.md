# NSAD-Bench

A benchmark that places symbolic and learned activation discovery under a
single training protocol on tabular binary classification: eight datasets,
ten random seeds, and fifteen activation configurations from four families.

Paper: *NSAD-Bench: A Reproducibility Benchmark for Symbolic and Learned
Activation Discovery in Tabular Classification*, Anas Hajbi, arXiv:FILLME.

This repository contains the benchmark code, the result files behind every
table and figure in the paper, and the figures themselves. Everything below
is what you need to reproduce any of it from scratch, or to evaluate a new
activation against the existing leaderboard.

## What it found

- **Padé Activation Units are easy to misconfigure, silently.** Initialized
  to the identity and trained in a single optimizer group, PAU is the weakest
  of the fourteen configurations evaluated. With the Leaky-ReLU
  initialization and separate coefficient optimizer specified by
  [Molina et al. (2020)](https://openreview.net/forum?id=BJlBSkHtDS), it
  improves on all eight datasets, by +0.031 to +0.410 AUC, mean +0.261,
  winning on all ten seeds of every dataset. Nothing in a results table
  distinguishes the two configurations.
- **Symbolic activations can fail completely.** On EEG Eye State and Wilt the
  discovered closed forms reach chance (0.500 and 0.503 AUC). Learned
  activations under the identical protocol reach 0.876 and 0.995.
- **Width beats activation choice here.** A ReLU network with roughly six
  times the parameters wins seven of eight datasets, so parameter-matched
  comparison is essential.
- **Discovered activations carry no dataset-specific structure.** Across a
  five-by-five transfer matrix the specialist activation is the best choice
  for its own dataset in three of five cases, and the diagonal exceeds the
  off-diagonal mean by 0.011 AUC.

---

## Contents

```
.
├── README.md                    this file
├── LICENSE
├── code/
│   ├── nsad_bench_main.py       symbolic discovery and main benchmark
│   ├── nsad_extra_methods.py    KAN, PAU-fixed, NAS-AF
│   └── make_main_figures.py     Figures 1 to 3
├── results/
│   ├── nsad_bench_results.csv   standard, PAU and symbolic rows of Table 2
│   ├── extra_methods.csv        KAN, PAU-fixed and NAS-AF rows of Table 2
│   ├── raw_per_seed_results.csv per-seed AUC for the standard, PAU and symbolic methods
│   ├── per_seed.csv             per-seed AUC for KAN, PAU-fixed and NAS-AF
│   ├── leaderboard.csv          merged leaderboard behind Table 2
│   ├── transfer_matrix_detail.csv   Section 4.5 and Figure 4
│   ├── spectral_cosine.csv      Section 4.5 and Figure 5
│   ├── sr_stochasticity.csv     Figure 6
│   └── discovered_formulas.json expression selected per dataset, Appendix C
└── figures/
    ├── fig1_rescue.png          Figure 1 (fig1_rescue.pdf is the vector version)
    ├── fig2_leaderboard.png     Figure 2 (fig2_leaderboard.pdf likewise)
    ├── fig3_family_summary.png  Figure 3 (fig3_family_summary.pdf likewise)
    ├── transfer_heatmap.png     Figure 4
    ├── spectral_full.png        Figure 5
    └── sr_stochasticity.png     Figure 6
```

Everything is produced by three scripts, run in order. Each writes to its own
timestamped folder, and each later step consumes the CSVs from the earlier
ones.

```
code/nsad_bench_main.py      ->  results_<timestamp>/          (step 1)
code/nsad_extra_methods.py   ->  results_extra_<timestamp>/    (step 2)
code/make_main_figures.py    ->  figures/                      (step 3)
```

You do not need to rerun step 1 to add a new activation to the leaderboard.
Step 1 is the expensive one, and its outputs are already in `results/`, so a
new method only needs steps 2 and 3.

---

## 0. Environment

Tested on Python 3.10 with a single NVIDIA RTX 3060 laptop GPU (6 GB) on
Linux. A CPU-only machine works but is far slower.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch numpy pandas scikit-learn scipy sympy matplotlib requests
```

Step 1 additionally needs PySR, which runs on Julia:

```bash
pip install juliacall pysr
python -c "import pysr; pysr.install()"   # first run downloads Julia deps
```

Steps 2 and 3 do **not** need Julia or PySR.

Optional: `pip install torchinfo` gives exact parameter counts in step 1.
Without it the script falls back to summing tensor sizes, which returns the
same numbers for these models.

### Import order

`juliacall` must be imported **before** `torch`, or the process can segfault.
`nsad_bench_main.py` already does this at the top of the file. If you
reorganise the imports, keep that order.

---

## 1. Main benchmark

```bash
cd code
python nsad_bench_main.py              # full run
python nsad_bench_main.py --medium     # 3 seeds, 50 epochs
python nsad_bench_main.py --quick      # 1 seed, 10 epochs, ~30 min
```

| Mode | Seeds | Epochs | PySR iters | Approximate wall time |
|---|---|---|---|---|
| `--quick` | 1 | 10 | 5 | 30 minutes |
| `--medium` | 3 | 50 | 100 | 8 hours |
| full (default) | 10 | 100 | 400 | ~240 GPU-hours, dominated by PySR |

Use `--quick` to confirm the environment works before committing to a full
run. Quick and medium modes are for sanity checks; only the full mode
reproduces the published numbers.

### Data

All eight datasets download automatically on first use, from OpenML and UCI.
HIGGS is the exception worth planning for: a 2.6 GB compressed download that
expands to roughly 8 GB as `HIGGS.csv` in the working directory, cached there
for later runs.

### What it produces

Into `results_<timestamp>/`:

| File | Used for |
|---|---|
| `nsad_bench_results.csv` | Table 2 (standard activations, PAU, symbolic rows) |
| `raw_per_seed_results.csv` | per-seed values behind Table 2, and the paired tests in Tables 3 and 4 |
| `transfer_matrix_detail.csv`, `transfer_matrix.csv` | Section 4.5, Figure 4 |
| `spectral_cosine.csv`, `spectral_kl.csv` | Section 4.5, Figure 5 |
| `sr_stochasticity.csv` | Figure 6 |
| `discovered_formulas.json` | the discovered expression per dataset |
| `dataset_summary.csv` | Table 1 |
| `significance_tests.csv` | paired tests |
| `abstraction_correlations.csv` | the univariate reduction check |
| `config.json`, `benchmark_log.txt` | the exact settings and full log |
| various `.png` / `.pdf` | diagnostic plots |

It also copies itself to `benchmark_script.py` inside the output folder, so
every results directory carries the code that produced it.

---

## 2. Learned activations (KAN, PAU-fixed, NAS-AF)

Point this at the `nsad_bench_results.csv` from step 1 so it can merge the two
halves into one leaderboard.

```bash
python nsad_extra_methods.py --main-csv ../results/nsad_bench_results.csv
```

| Mode | Seeds | Epochs | Wall time on an RTX 3060 |
|---|---|---|---|
| `--quick` | 1 | 10 | ~10 minutes |
| `--medium` | 3 | 50 | ~5 hours |
| full (default) | 10 | 100 | 15 to 18 hours |

Into `results_extra_<timestamp>/`:

| File | Used for |
|---|---|
| `extra_methods.csv` | the KAN, PAU-fixed and NAS-AF rows of Table 2 |
| `per_seed.csv` | per-seed values for those three methods |
| `leaderboard.csv` | step 1 and step 2 merged |

KAN is the slow one here: the B-spline forward pass adds roughly thirty
percent over ReLU.

---

## 3. Figures

```bash
python make_main_figures.py \
    --main  ../results/nsad_bench_results.csv \
    --extra ../results/extra_methods.csv \
    --out   ../figures
```

Produces `fig1_rescue`, `fig2_leaderboard` and `fig3_family_summary`, each as
both PNG and PDF, and prints suggested captions with the numbers filled in.
Figures 4 to 6 come directly from step 1.

---

## Adding your own activation

Two edits in `nsad_extra_methods.py`:

1. Write an `nn.Module` whose `forward` maps a tensor to a tensor of the same
   shape, with any learnable parameters registered as `nn.Parameter`.
2. Add one line to the `methods` registry inside `main()`:

```python
methods = OrderedDict([
    ("KAN",       lambda: KANActivation(...)),
    ("PAU-fixed", lambda: PadeActivationUnitFixed(m=5, n=4)),
    ("NAS-AF",    lambda: SwishLearnableBeta(beta_init=1.0)),
    ("YourMethod", lambda: YourActivation()),      # <- add this
])
```

The registry takes a factory rather than an instance, because each layer needs
its own independent parameters. Then run steps 2 and 3. Your method is trained
under exactly the protocol every other row used, with no hyperparameters
chosen in its favour.

If your activation needs different optimizer treatment, extend
`make_optimizer`, and say so when you report the result. PAU-fixed is the one
row in the paper that does this, and the paper flags it wherever it appears.

---

## Compute

Every experiment reported in the paper was run on a single NVIDIA RTX 3060
laptop GPU with 6 GB of memory.

| Stage | Cost |
|---|---|
| Symbolic discovery and main benchmark | roughly 240 GPU-hours, dominated by symbolic regression rather than network training |
| The three learned activations | 15 to 18 hours |
| Figures | seconds |

Evaluating an additional activation requires only the second stage, since the
discovered expressions are released as fixed artifacts.

---

## Reproducibility notes

Seeds are set in `random`, `numpy` and `torch`, with
`cudnn.deterministic = True` and `cudnn.benchmark = False`. PySR runs with
`deterministic=True`, `procs=0` and `multithreading=False`. Repeated runs on
the same machine and library versions reproduce the released numbers; results
across different GPUs or CUDA versions can differ in the last decimal place.

Two details of the protocol worth knowing before you compare against it:

- The inner train/validation split uses a fixed `random_state=42` for every
  seed, so seed variation covers the outer train/test split and weight
  initialisation, not the validation partition.
- Standard deviations are computed with `np.std`, that is, the population form
  with no Bessel correction.

---

## Checking a number from the paper

Every number traces back to a released CSV. The KAN entry for EEG Eye State in
Table 2:

```python
import pandas as pd
df = pd.read_csv('results/per_seed.csv')
sel = df[(df.Dataset == 'EEG_EYE_STATE') & (df.Model == 'KAN')]
print(sel.AUC.mean(), sel.AUC.std(ddof=0), len(sel))
```

The transfer numbers in Section 4.5:

```python
t = pd.read_csv('results/transfer_matrix_detail.csv')
piv = t.pivot(index='Source', columns='Target', values='AUC_mean')
print(piv.round(3))
```

---

## Troubleshooting

**Segfault on start.** `juliacall` was imported after `torch`. Keep the import
order at the top of `nsad_bench_main.py`.

**PySR fails or Julia is missing.** `nsad_bench_main.py` detects this and falls
back to three hard-coded formulas so that the rest of the pipeline still runs.
Those fallback results are not the published numbers, and the log says so.
Install `juliacall` and `pysr` for a real run.

**Out of memory.** Lower `BATCH_SIZE` in the mode you are running, or reduce
`HIGGS_ROWS`. Everything else fits comfortably in 6 GB.

**HIGGS download stalls.** Fetch
`https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz`
by hand, decompress it to `HIGGS.csv`, and put it in the working directory.
The script skips the download if that file exists.

**OpenML timeouts.** Usually transient; rerun. Datasets are cached by
scikit-learn after the first successful fetch.

---

## Citation

```bibtex
@article{hajbi2026nsadbench,
  title   = {{NSAD-Bench}: A Reproducibility Benchmark for Symbolic and
             Learned Activation Discovery in Tabular Classification},
  author  = {Hajbi, Anas},
  journal = {arXiv preprint arXiv:FILLME},
  year    = {2026}
}
```

## License

Code under the MIT License, see `LICENSE`. The paper text and figures are
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
