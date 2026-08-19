"""
==========================================================================
NSAD-Bench: ADDITIONAL ACTIVATION-DISCOVERY METHODS
==========================================================================
Adds three learned-activation baselines on top of the main benchmark
run produced by nsad_bench_main.py:

  1. KAN-style learnable B-spline activation (drop-in for LightMLP).
     Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024).
  2. PAU re-run with the original Molina et al. (ICLR 2020) hyperparameters
     (separate optimizer group, lower learning rate, weight decay).
  3. NAS-AF baseline: Swish-style learnable family f(x) = x * sigmoid(beta*x).
     Reference: Ramachandran et al., "Searching for Activation Functions" (2017).

PROTOCOL: identical to the main run -- 10 seeds (42-51), 100 epochs with early
stopping patience 10, 2-layer LightMLP (64->32), BCE-with-logits,
batch 2048, Adam lr=1e-3, 80/20 train/test split, 90/10 train/val,
StandardScaler fit on training only.

This script does NOT touch:
  * SR-discovered activations (produced by nsad_bench_main.py)
  * Standard baselines (ReLU/GELU/SiLU/...) (produced by nsad_bench_main.py)
  * Transfer matrix, spectral analysis, depth ablation, manual search

Output:
  results_<timestamp>/extra_methods.csv  -- new methods, same schema as the main run
  results_<timestamp>/per_seed.csv       -- per-seed AUCs
  results_<timestamp>/leaderboard.csv    -- main run + new methods, merged

Usage:
    python nsad_extra_methods.py --main-csv path/to/nsad_bench_results.csv

Compute: ~15-18 hours on a single GPU for full mode.
==========================================================================
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_covtype, fetch_openml
import requests, gzip, shutil, os, random, sys, warnings, time, json
import copy
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings("ignore")


# ==========================================================================
# CONFIGURATION
# ==========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="quick mode: 1 seed, 2 datasets, 10 epochs")
parser.add_argument("--medium", action="store_true",
                    help="medium mode: 3 seeds, all datasets, 50 epochs")
parser.add_argument("--main-csv", dest="main_csv", type=str,
                    default=None,
                    help="path to nsad_bench_results.csv from the main benchmark "
                         "run, used to merge both halves into one leaderboard")
args = parser.parse_args()

if args.quick:
    SEEDS = [42]
    EPOCHS = 10
    PATIENCE = 3
    BATCH_SIZE = 512
    HIGGS_ROWS = 5000
    DATASETS_TO_RUN = ["FOREST_COVER", "EEG_EYE_STATE"]
elif args.medium:
    SEEDS = [42, 43, 44]
    EPOCHS = 50
    PATIENCE = 7
    BATCH_SIZE = 1024
    HIGGS_ROWS = 100000
    DATASETS_TO_RUN = None  # all
else:
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    EPOCHS = 100
    PATIENCE = 10
    BATCH_SIZE = 2048
    HIGGS_ROWS = 500000
    DATASETS_TO_RUN = None  # all

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path(f"results_extra_{TIMESTAMP}")
OUT.mkdir(exist_ok=True)


def out_path(fn):
    return str(OUT / fn)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# All 8 datasets, identical config to the main run
DATASETS_CONFIG = OrderedDict([
    ("HIGGS",           {"limit": HIGGS_ROWS}),
    ("FOREST_COVER",    {"limit": 100000}),
    ("SPAMBASE",        {"limit": None}),
    ("MINIBOONE",       {"limit": 100000}),
    ("MAGIC_TELESCOPE", {"limit": None}),
    ("PHONEME",         {"limit": None}),
    ("EEG_EYE_STATE",   {"limit": None}),
    ("WILT",            {"limit": None}),
])
if DATASETS_TO_RUN is not None:
    DATASETS_CONFIG = OrderedDict(
        [(k, v) for k, v in DATASETS_CONFIG.items() if k in DATASETS_TO_RUN])


print(f"{'=' * 70}")
print(f"  NSAD-Bench: ADDITIONAL METHODS (KAN, PAU-fixed, NAS-AF)")
print(f"  Seeds: {len(SEEDS)} | Epochs: {EPOCHS} | Device: {DEVICE}")
print(f"  Datasets: {list(DATASETS_CONFIG.keys())}")
print(f"  Output: {OUT}")
print(f"{'=' * 70}\n")


# ==========================================================================
# DATA LOADING (identical to the main run)
# ==========================================================================
def get_dataset(name, limit_rows=None):
    if name == "HIGGS":
        if not os.path.exists("HIGGS.csv"):
            print("    Downloading HIGGS (~2.6 GB compressed)...")
            url = ("https://archive.ics.uci.edu/ml/machine-learning-"
                   "databases/00280/HIGGS.csv.gz")
            with requests.get(url, stream=True) as r:
                with open("HIGGS.csv.gz", "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            with gzip.open("HIGGS.csv.gz", "rb") as fi:
                with open("HIGGS.csv", "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            if os.path.exists("HIGGS.csv.gz"):
                os.remove("HIGGS.csv.gz")
        rows = limit_rows or 500000
        df = pd.read_csv("HIGGS.csv", header=None, nrows=rows)
        y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values
    elif name == "FOREST_COVER":
        data = fetch_covtype()
        X, y = data.data, (data.target == 2).astype(int)
        if limit_rows and len(y) > limit_rows:
            idx = np.random.choice(len(y), limit_rows, replace=False)
            X, y = X[idx], y[idx]
    elif name == "SPAMBASE":
        data = fetch_openml(name="spambase", version=1, as_frame=False)
        X, y = data.data, data.target.astype(int)
    elif name == "MINIBOONE":
        data = fetch_openml(data_id=41150, as_frame=False)
        X = data.data.astype(float)
        t = data.target
        if t.dtype == object:
            t = LabelEncoder().fit_transform(t)
        y = t.astype(int)
        if limit_rows and len(y) > limit_rows:
            idx = np.random.choice(len(y), limit_rows, replace=False)
            X, y = X[idx], y[idx]
    elif name == "MAGIC_TELESCOPE":
        data = fetch_openml(name="MagicTelescope", version=1, as_frame=False)
        X = data.data.astype(float)
        t = data.target
        if t.dtype == object:
            y = (t == "g").astype(int)
        else:
            y = t.astype(int)
    elif name == "PHONEME":
        data = fetch_openml(name="phoneme", version=1, as_frame=False)
        X = data.data.astype(float)
        t = data.target
        if t.dtype == object:
            t = LabelEncoder().fit_transform(t)
        y = t.astype(int)
    elif name == "EEG_EYE_STATE":
        data = fetch_openml(data_id=1471, as_frame=False)
        X = data.data.astype(float)
        t = data.target
        if t.dtype == object:
            t = LabelEncoder().fit_transform(t)
        y = t.astype(int)
    elif name == "WILT":
        data = fetch_openml(data_id=40983, as_frame=False)
        X = data.data.astype(float)
        t = data.target
        if t.dtype == object:
            t = LabelEncoder().fit_transform(t)
        y = t.astype(int)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]


# ==========================================================================
# NEW ACTIVATIONS
# ==========================================================================

# ---- 1. KAN-style learnable B-spline activation ----
class KANActivation(nn.Module):
    """Drop-in KAN-style activation. Uses a sum of learnable B-spline
    basis functions plus a residual base activation (SiLU as in the
    original KAN paper). See Liu et al. (2024) Sec 2.

    f(x) = w_b * silu(x) + w_s * sum_i c_i * B_i(x)

    Spline grid is fixed at init time over [-grid_range, grid_range],
    grid is uniform with grid_size + 2k + 1 knots (k = spline order).
    Coefficients c_i are learnable; w_b and w_s are learnable scalars.

    For an MLP-with-fixed-architecture (as in our protocol), this gives
    one independent learnable spline activation per layer instance.
    """
    def __init__(self, grid_size=5, spline_order=3,
                 grid_range=(-3.0, 3.0)):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        # Fixed grid (not learnable to keep protocol clean and avoid
        # PyKAN's grid-update complexity).
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(
            -spline_order, grid_size + spline_order + 1
        ) * h + grid_range[0]
        self.register_buffer("grid", grid)  # (grid_size + 2k + 1,)

        n_basis = grid_size + spline_order  # number of B-spline basis
        self.coeffs = nn.Parameter(torch.zeros(n_basis))
        self.w_base = nn.Parameter(torch.tensor(1.0))
        self.w_spline = nn.Parameter(torch.tensor(1.0))

        # Initialize spline coefficients with small noise so different
        # layer instances diverge during training.
        with torch.no_grad():
            self.coeffs.copy_(0.1 * torch.randn(n_basis))

    def b_splines(self, x):
        """Compute B-spline basis values for input x.
        Returns (B, n_basis) tensor where B = x.numel().
        """
        # Flatten x for vectorized basis computation
        x_flat = x.reshape(-1, 1)             # (B, 1)
        grid = self.grid.unsqueeze(0)         # (1, n_grid)
        # Order-0 indicators: 1 if grid[i] <= x < grid[i+1] else 0
        bases = ((x_flat >= grid[:, :-1]) &
                 (x_flat < grid[:, 1:])).float()  # (B, n_grid - 1)
        # Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            left = (x_flat - grid[:, : -(k + 1)]) / (
                grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)
            right = (grid[:, k + 1:] - x_flat) / (
                grid[:, k + 1:] - grid[:, 1:-k] + 1e-8)
            bases = left * bases[:, :-1] + right * bases[:, 1:]
        return bases  # (B, n_basis)

    def forward(self, x):
        original_shape = x.shape
        bases = self.b_splines(x)            # (B, n_basis)
        spline_out = bases @ self.coeffs     # (B,)
        spline_out = spline_out.reshape(original_shape)
        base_out = F.silu(x)
        return self.w_base * base_out + self.w_spline * spline_out


# ---- 2. PAU re-run with Molina et al.'s recommended settings ----
class PadeActivationUnitFixed(nn.Module):
    """PAU with Molina et al.'s ICLR 2020 recommended setup:
      * (m, n) = (5, 4), same as the main run.
      * Initialize a, b coefficients to approximate Leaky ReLU instead of
        identity. This is the key change: the main run used identity init.
      * Coefficients tagged with `_pau_param=True` so we can apply a
        separate optimizer (lower lr) at training time.

    Leaky-ReLU init values from Table 6 of Molina et al. (alpha=0.01,
    safe denominator):
      a = [0.0218, 0.5, 0.5907, 0, -0.0468, 0]   (numerator, degree 5)
      b = [0, 0.0936, 0, 0.0156]                 (denominator, degree 4)
    """
    LEAKY_RELU_INIT_A = [0.0218, 0.5, 0.5907, 0.0, -0.0468, 0.0]
    LEAKY_RELU_INIT_B = [0.0, 0.0936, 0.0, 0.0156]

    def __init__(self, m=5, n=4):
        super().__init__()
        self.m, self.n = m, n
        self.a = nn.Parameter(torch.tensor(
            self.LEAKY_RELU_INIT_A[: m + 1], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(
            self.LEAKY_RELU_INIT_B[: n], dtype=torch.float32))
        # Tag so the optimizer can distinguish PAU params
        self.a._pau_param = True
        self.b._pau_param = True

    def forward(self, x):
        P = self.a[0] * torch.ones_like(x)
        x_pow = x.clone()
        for i in range(1, self.m + 1):
            P = P + self.a[i] * x_pow
            if i < self.m:
                x_pow = x_pow * x
        Q = torch.ones_like(x)
        x_abs_pow = x.abs()
        for j in range(self.n):
            Q = Q + self.b[j].abs() * x_abs_pow
            if j < self.n - 1:
                x_abs_pow = x_abs_pow * x.abs()
        return P / (Q + 1e-8)


# ---- 3. NAS-AF: Swish-style learnable family ----
class SwishLearnableBeta(nn.Module):
    """f(x) = x * sigmoid(beta * x), with learnable beta.
    This is the activation Ramachandran et al. (2017) "discovered" via
    RL search. Using it as a baseline answers: 'when you compare your
    SR-discovered activation to a NAS-discovered learnable family, who
    wins?'"""
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# ==========================================================================
# MODEL: identical to the main run's LightMLP (depth 2, widths 64->32)
# ==========================================================================
class LightMLP(nn.Module):
    def __init__(self, input_dim, activation_factory):
        super().__init__()
        # Use deepcopy via factory to ensure each layer gets its own
        # independent learnable parameters (KAN, PAU, Swish all have
        # learnable parameters per instance).
        self.lin1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = activation_factory()
        self.drop1 = nn.Dropout(0.3)
        self.lin2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = activation_factory()
        self.drop2 = nn.Dropout(0.3)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.act1(self.bn1(self.lin1(x)))
        h = self.drop1(h)
        h = self.act2(self.bn2(self.lin2(h)))
        h = self.drop2(h)
        return self.head(h)


# ==========================================================================
# OPTIMIZER (special-cased for PAU)
# ==========================================================================
def make_optimizer(model, method_name):
    """For PAU-fixed: use lower lr (1e-4) and weight decay 5e-4 on
    PAU coefficients, per Molina et al. recommendation. Other params
    use Adam lr=1e-3 (protocol default).

    For all other methods: standard Adam lr=1e-3 across all params
    (matches the main protocol exactly)."""
    if method_name == "PAU-fixed":
        pau_params = [p for p in model.parameters()
                      if getattr(p, "_pau_param", False)]
        other_params = [p for p in model.parameters()
                        if not getattr(p, "_pau_param", False)]
        return torch.optim.Adam([
            {"params": other_params, "lr": 1e-3, "weight_decay": 0.0},
            {"params": pau_params, "lr": 1e-4, "weight_decay": 5e-4},
        ])
    return torch.optim.Adam(model.parameters(), lr=1e-3)


# ==========================================================================
# TRAINING (identical schedule to the main run)
# ==========================================================================
def train_and_eval(model, X_tr, y_tr, X_te, y_te, method_name):
    model.to(DEVICE)
    opt = make_optimizer(model, method_name)
    crit = nn.BCEWithLogitsLoss()

    X_tr2, X_v, y_tr2, y_v = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=42)
    ds = TensorDataset(
        torch.tensor(X_tr2, dtype=torch.float32),
        torch.tensor(y_tr2, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    X_v_t = torch.tensor(X_v, dtype=torch.float32).to(DEVICE)
    y_v_t = torch.tensor(y_v, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    best_v, patience, best_state, ep_used = float("inf"), 0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(X_v_t), y_v_t).item()
        if v < best_v:
            best_v, patience = v, 0
            best_state = copy.deepcopy(model.state_dict())
            ep_used = ep + 1
        else:
            patience += 1
        if patience >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        probs = torch.sigmoid(model(xt)).cpu().numpy()
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_te, preds)
    try:
        auc = roc_auc_score(y_te, probs)
    except ValueError:
        auc = 0.5
    f1 = f1_score(y_te, preds, zero_division=0)
    return acc, auc, f1, ep_used


def run_method_on_dataset(method_name, activation_factory,
                          X, y, input_dim):
    """Run a method across all SEEDS on one dataset. Returns dict of
    aggregated stats + per-seed arrays."""
    accs, aucs, f1s, eps = [], [], [], []
    n_params = None
    for seed in SEEDS:
        set_seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        model = LightMLP(input_dim, activation_factory)
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
        acc, auc, f1, ep = train_and_eval(
            model, X_tr, y_tr, X_te, y_te, method_name)
        accs.append(acc)
        aucs.append(auc)
        f1s.append(f1)
        eps.append(ep)
    return {
        "method": method_name,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "params": n_params,
        "epochs_avg": float(np.mean(eps)),
        "aucs": aucs, "accs": accs, "f1s": f1s, "eps": eps,
    }


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    # Method registry: (display name, activation factory)
    methods = OrderedDict([
        ("KAN",       lambda: KANActivation(grid_size=5, spline_order=3,
                                              grid_range=(-3.0, 3.0))),
        ("PAU-fixed", lambda: PadeActivationUnitFixed(m=5, n=4)),
        ("NAS-AF",    lambda: SwishLearnableBeta(beta_init=1.0)),
    ])

    # Load all datasets first
    print(f"Loading {len(DATASETS_CONFIG)} datasets...\n")
    dataset_cache = {}
    for dname, cfg in DATASETS_CONFIG.items():
        try:
            set_seed(42)
            X, y = get_dataset(dname, limit_rows=cfg["limit"])
            dataset_cache[dname] = (X, y)
            print(f"  OK {dname}: shape={X.shape}, "
                  f"pos_rate={y.mean():.3f}")
        except Exception as e:
            print(f"  FAIL {dname}: {e}")
    print()

    # Run all (method, dataset) pairs
    summary_rows = []
    per_seed_rows = []

    total_runs = len(methods) * len(dataset_cache)
    run_idx = 0

    for method_name, factory in methods.items():
        print(f"\n{'=' * 70}")
        print(f"  METHOD: {method_name}")
        print(f"{'=' * 70}")
        for dname, (X, y) in dataset_cache.items():
            run_idx += 1
            t0 = time.time()
            print(f"\n  [{run_idx}/{total_runs}] {method_name} on {dname}...")
            try:
                r = run_method_on_dataset(
                    method_name, factory, X, y, X.shape[1])
                elapsed = time.time() - t0
                print(f"    AUC = {r['auc_mean']:.4f} +/- {r['auc_std']:.4f}  "
                      f"({elapsed/60:.1f} min, {r['epochs_avg']:.0f} epochs avg)")

                summary_rows.append({
                    "Dataset": dname,
                    "Model": method_name,
                    "Experiment": "Main",
                    "AUC_mean": r["auc_mean"],
                    "AUC_std": r["auc_std"],
                    "Acc_mean": r["acc_mean"],
                    "Acc_std": r["acc_std"],
                    "F1_mean": r["f1_mean"],
                    "F1_std": r["f1_std"],
                    "Params": r["params"],
                    "Epochs_avg": r["epochs_avg"],
                    "Source": "extra_methods",
                })

                for seed, auc, acc, f1 in zip(
                        SEEDS, r["aucs"], r["accs"], r["f1s"]):
                    per_seed_rows.append({
                        "Dataset": dname,
                        "Model": method_name,
                        "Seed": seed,
                        "AUC": auc, "Acc": acc, "F1": f1,
                    })
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Save
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path("extra_methods.csv"), index=False)
    pd.DataFrame(per_seed_rows).to_csv(
        out_path("per_seed.csv"), index=False)
    print(f"\n  Saved: extra_methods.csv  ({len(summary_rows)} rows)")
    print(f"  Saved: per_seed.csv  ({len(per_seed_rows)} rows)")

    # Merge with the main run if provided
    leaderboard = None
    if args.main_csv and os.path.exists(args.main_csv):
        print(f"\n  Merging with main-run results from {args.main_csv}...")
        main_df = pd.read_csv(args.main_csv)
        # Take only the Main experiment from the main run
        main_only = main_df[main_df["Experiment"] == "Main"].copy()
        main_only["Source"] = "main_run"
        # Drop columns we don't have or don't need for leaderboard
        keep_cols = ["Dataset", "Model", "Experiment",
                     "AUC_mean", "AUC_std",
                     "Acc_mean", "Acc_std",
                     "F1_mean", "F1_std", "Params", "Source"]
        main_keep = main_only[[c for c in keep_cols if c in main_only.columns]]
        new_keep = summary_df[[c for c in keep_cols if c in summary_df.columns]]
        leaderboard = pd.concat([main_keep, new_keep], ignore_index=True)
        leaderboard.to_csv(out_path("leaderboard.csv"), index=False)
        print(f"  Saved: leaderboard.csv  ({len(leaderboard)} rows)")
    else:
        if args.main_csv:
            print(f"\n  [WARN] main-run CSV '{args.main_csv}' not found. "
                  f"Skipping merge.")
        else:
            print(f"\n  No main-run CSV provided (--main-csv). "
                  f"Skipping leaderboard merge.")
        # Still save the new methods as a standalone leaderboard
        summary_df.to_csv(out_path("leaderboard.csv"), index=False)

    # Pretty-print summary table
    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}\n")
    if leaderboard is not None:
        # Pivot: dataset rows, method columns, AUC values
        try:
            pivot = leaderboard.pivot_table(
                index="Dataset", columns="Model",
                values="AUC_mean", aggfunc="mean")
            print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
        except Exception as e:
            print(f"  [could not pivot: {e}]")
            print(summary_df.to_string(index=False))
    else:
        for dname in dataset_cache:
            ds_rows = [r for r in summary_rows if r["Dataset"] == dname]
            print(f"  {dname}:")
            for r in ds_rows:
                print(f"    {r['Model']:<12} AUC = {r['AUC_mean']:.4f} "
                      f"+/- {r['AUC_std']:.4f}")

    print(f"\n  Outputs in: {OUT}/")


if __name__ == "__main__":
    t_start = time.time()
    main()
    elapsed_h = (time.time() - t_start) / 3600
    print(f"\n  Wall time: {elapsed_h:.1f} hours\n")
