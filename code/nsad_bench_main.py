"""
==========================================================================
NSAD-BENCH — COMPLETE BENCHMARK
==========================================================================
Main benchmark run. Produces every result in the paper except the three
learned activations added in the second sweep (KAN, PAU-fixed, NAS-AF),
which are produced by nsad_extra_methods.py.

This script covers:
  * PySR symbolic discovery on all 8 datasets, with a discovery-fraction
    sweep on the first three
  * Main benchmark: Heavy ReLU, 7 standard activations, PAU (naive
    configuration), Hybrid (Specialist), Unbounded Hybrid, and
    Hybrid (Constrained) on two datasets
  * Full 5x5 transfer matrix
  * Spectral analysis (FFT magnitude, cosine similarity, KL divergence)
  * Discovery stochasticity: N_SR_RUNS independent PySR runs per dataset
  * Manual-search ablation over 11 hand-chosen activations
  * Depth ablation and component ablation
  * Abstraction correlation between the multivariate formula and its
    univariate elementwise reduction

ALL outputs (CSVs, PNGs, PDFs, JSONs) are written to a timestamped folder:
    results_YYYYMMDD_HHMMSS/

Usage:
    python nsad_bench_main.py              # Full run
    python nsad_bench_main.py --medium     # Medium (~8h)
    python nsad_bench_main.py --quick      # Quick test (~30min)
==========================================================================
"""

# CRITICAL: juliacall must be imported BEFORE torch to avoid segfault
try:
    import juliacall  # noqa: F401
    HAS_JULIA = True
except ImportError:
    HAS_JULIA = False
    print("[WARNING] juliacall not found. PySR discovery will be skipped.")
    print("          Install: pip install juliacall pysr")

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
import sympy
from sympy import Function as SympyFunction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.fft import fft, fftfreq
import requests, gzip, shutil, os, re, random, sys, warnings, time, json
import copy
from datetime import datetime
from collections import OrderedDict
from pathlib import Path

warnings.filterwarnings("ignore")

if HAS_JULIA:
    from pysr import PySRRegressor


# ==========================================================================
# 0. OUTPUT DIRECTORY WITH TIMESTAMP
# ==========================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results_{TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def out_path(filename):
    """Return full path inside the timestamped output folder."""
    return str(OUTPUT_DIR / filename)

# Also save a copy of this script for reproducibility
try:
    import shutil as _sh
    _sh.copy2(__file__, out_path("benchmark_script.py"))
except Exception:
    pass


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================
MODE = "full"
if "--quick" in sys.argv:
    MODE = "quick"
elif "--medium" in sys.argv:
    MODE = "medium"

CONFIG = {
    "quick": dict(
        SEEDS=[42], HIGGS_ROWS=5000, DISCOVERY_ROWS=2000,
        EPOCHS=10, PATIENCE=3, BATCH_SIZE=512,
        DATA_FRACTIONS=[0.2, 1.0],
        PYSR_ITERS=5, PYSR_POPS=5, PYSR_POP_SIZE=20,
        N_SR_RUNS=1, N_MANUAL_SEEDS=1,
        TRANSFER_DATASETS=["HIGGS", "FOREST_COVER", "SPAMBASE"],
        DEPTH_DATASETS=["FOREST_COVER"],
        ABLATION_DATASETS=["FOREST_COVER"],
    ),
    "medium": dict(
        SEEDS=[42, 43, 44], HIGGS_ROWS=100000, DISCOVERY_ROWS=20000,
        EPOCHS=50, PATIENCE=7, BATCH_SIZE=1024,
        DATA_FRACTIONS=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        PYSR_ITERS=100, PYSR_POPS=20, PYSR_POP_SIZE=30,
        N_SR_RUNS=3, N_MANUAL_SEEDS=3,
        TRANSFER_DATASETS=["HIGGS", "FOREST_COVER", "SPAMBASE",
                           "MINIBOONE", "MAGIC_TELESCOPE"],
        DEPTH_DATASETS=["HIGGS", "FOREST_COVER"],
        ABLATION_DATASETS=["HIGGS", "FOREST_COVER"],
    ),
    "full": dict(
        SEEDS=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        HIGGS_ROWS=500000, DISCOVERY_ROWS=50000,
        EPOCHS=100, PATIENCE=10, BATCH_SIZE=2048,
        DATA_FRACTIONS=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        PYSR_ITERS=400, PYSR_POPS=50, PYSR_POP_SIZE=50,
        N_SR_RUNS=5, N_MANUAL_SEEDS=10,
        TRANSFER_DATASETS=["HIGGS", "FOREST_COVER", "SPAMBASE",
                           "MINIBOONE", "MAGIC_TELESCOPE"],
        DEPTH_DATASETS=["HIGGS", "FOREST_COVER", "MINIBOONE"],
        ABLATION_DATASETS=["HIGGS", "FOREST_COVER"],
    ),
}[MODE]

# Unpack config into globals
for _k, _v in CONFIG.items():
    globals()[_k] = _v

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAC_LABEL = lambda f: f"{int(f * 100)}pct"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# All 8 datasets
DATASETS_CONFIG = OrderedDict([
    ("HIGGS",           {"limit": HIGGS_ROWS, "domain": "physics",   "geometry": "continuous"}),
    ("FOREST_COVER",    {"limit": 100000,     "domain": "ecology",   "geometry": "continuous"}),
    ("SPAMBASE",        {"limit": None,       "domain": "text",      "geometry": "discrete"}),
    ("MINIBOONE",       {"limit": 100000,     "domain": "physics",   "geometry": "continuous"}),
    ("MAGIC_TELESCOPE", {"limit": None,       "domain": "astrophys", "geometry": "continuous"}),
    ("PHONEME",         {"limit": None,       "domain": "signal",    "geometry": "continuous"}),
    ("EEG_EYE_STATE",   {"limit": None,       "domain": "neuro",     "geometry": "continuous"}),
    ("WILT",            {"limit": None,       "domain": "remote_s",  "geometry": "continuous"}),
])

# Save config
with open(out_path("config.json"), "w") as _f:
    json.dump({
        "mode": MODE, "timestamp": TIMESTAMP, "device": str(DEVICE),
        **{k: v if not isinstance(v, list) else v for k, v in CONFIG.items()},
        "datasets": list(DATASETS_CONFIG.keys()),
    }, _f, indent=2)

print(f"{'=' * 70}")
print(f"  NSAD-BENCH — COMPLETE BENCHMARK")
print(f"  Mode: {MODE} | Device: {DEVICE} | Seeds: {len(SEEDS)}")
print(f"  Epochs: {EPOCHS} (Early Stop patience={PATIENCE})")
print(f"  Datasets: {len(DATASETS_CONFIG)} | SR runs/dataset: {N_SR_RUNS}")
print(f"  Output folder: {OUTPUT_DIR}")
print(f"{'=' * 70}")


# ==========================================================================
# 2. DATA LOADING — 8 DATASETS
# ==========================================================================
def get_dataset(name, limit_rows=None):
    """Load dataset. Returns X (np), y (binary np)."""
    if name == "HIGGS":
        if not os.path.exists("HIGGS.csv"):
            print("    Downloading HIGGS (~2.6 GB compressed)...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
            with requests.get(url, stream=True) as r:
                with open("HIGGS.csv.gz", 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            with gzip.open("HIGGS.csv.gz", 'rb') as fi:
                with open("HIGGS.csv", 'wb') as fo:
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
        data = fetch_openml(name='spambase', version=1, as_frame=False)
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
        data = fetch_openml(name='MagicTelescope', version=1, as_frame=False)
        X = data.data.astype(float)
        t = data.target
        if t.dtype == object:
            y = (t == 'g').astype(int)
        else:
            y = t.astype(int)

    elif name == "PHONEME":
        data = fetch_openml(name='phoneme', version=1, as_frame=False)
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

    # Remove NaN rows
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    return X, y


# ==========================================================================
# 3. FORMULA PARSING & COMPILATION
# ==========================================================================
class safe_sqrt(SympyFunction):
    @classmethod
    def eval(cls, x):
        return None

class square(SympyFunction):
    @classmethod
    def eval(cls, x):
        return None

class relu_sym(SympyFunction):
    @classmethod
    def eval(cls, x):
        return None

SYMPY_LOCALS = {
    "safe_sqrt": safe_sqrt, "square": square, "relu": relu_sym,
    "exp": sympy.exp, "cos": sympy.cos, "sin": sympy.sin,
    "atan": sympy.atan, "Abs": sympy.Abs, "sqrt": sympy.sqrt,
    "log": sympy.log, "Max": sympy.Max,
}


def parse_formula(formula_str):
    formula_str = re.sub(r'\s+', '', formula_str)
    try:
        return sympy.sympify(formula_str, locals=SYMPY_LOCALS)
    except Exception as e:
        print(f"    [WARN] sympify failed: {e}")
        return sympy.sympify("x * cos(x)", locals=SYMPY_LOCALS)


def sympy_to_torch_func(sympy_expr):
    """Convert sympy expression -> PyTorch activation (univariate abstraction)."""
    symbols_in_expr = list(sympy_expr.free_symbols)

    if len(symbols_in_expr) == 0:
        c = float(sympy_expr)
        return lambda x: torch.ones_like(x) * c, str(sympy_expr)

    x_sym = sympy.Symbol('x')
    generalized = sympy_expr
    for s in symbols_in_expr:
        generalized = generalized.subs(s, x_sym)
    gen_str = str(generalized)

    torch_map = {
        "sin": torch.sin, "cos": torch.cos, "atan": torch.atan,
        "tan": torch.tan, "Abs": torch.abs, "abs": torch.abs,
        "exp": lambda t: torch.exp(torch.clamp(t, max=50.0)),
        "log": lambda t: torch.log(torch.abs(t) + 1e-8),
        "sqrt": lambda t: torch.sqrt(torch.abs(t) + 1e-8),
        "safe_sqrt": lambda t: torch.sqrt(torch.abs(t) + 1e-8),
        "square": lambda t: t ** 2,
        "relu": lambda t: torch.relu(t),
        "Max": torch.max, "sign": torch.sign, "Pow": torch.pow,
    }

    try:
        raw_fn = sympy.lambdify(x_sym, generalized, modules=[torch_map, "numpy"])

        def activation_fn(x):
            out = raw_fn(x)
            if not isinstance(out, torch.Tensor):
                out = torch.full_like(x, float(out))
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
            return torch.clamp(out, -50.0, 50.0)

        # Smoke test
        test_out = activation_fn(torch.randn(10))
        assert not torch.isnan(test_out).any()
        return activation_fn, gen_str

    except Exception as e:
        print(f"    [WARN] lambdify failed for '{gen_str[:60]}': {e}")
        return lambda x: x * torch.cos(x), "x*cos(x) [fallback]"


class SymbolicActivation(nn.Module):
    def __init__(self, sympy_expr_or_str, label="discovered"):
        super().__init__()
        self.label = label
        if isinstance(sympy_expr_or_str, str):
            expr = parse_formula(sympy_expr_or_str)
        else:
            expr = sympy_expr_or_str
        self.torch_func, self.formula_str = sympy_to_torch_func(expr)

    def forward(self, x):
        return self.torch_func(x)

    def extra_repr(self):
        return f"formula={self.formula_str}"


# ==========================================================================
# 4. ACTIVATION FUNCTIONS
# ==========================================================================

class PadeActivationUnit(nn.Module):
    """
    Pade Activation Unit (Molina et al., ICLR 2020), NAIVE CONFIGURATION.
    Learnable rational activation: P(x)/Q(x).
    Safe denominator: Q(x) = 1 + sum |b_j| |x|^j  (always > 0).

    NOTE: this initializes to the identity and is trained in the single
    optimizer group used by every other method. The paper's PAU-fixed row
    (in nsad_extra_methods.py) uses the Leaky-ReLU initialization and the
    separate coefficient optimizer recommended by Molina et al.
    """
    def __init__(self, m=5, n=4):
        super().__init__()
        self.m, self.n = m, n
        self.a = nn.Parameter(torch.zeros(m + 1))
        self.b = nn.Parameter(torch.zeros(n))
        if m >= 1:
            self.a.data[1] = 1.0  # init ~ identity

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

    def extra_repr(self):
        return f"PAU(m={self.m}, n={self.n})"


class UnboundedHybridActivation(nn.Module):
    """
    Addresses the bounded-activation vanishing gradient problem.
    f(x) = ReLU(x) + alpha * sigma_discovered(x),  alpha learnable (init 0.1).
    """
    def __init__(self, discovered_expr, init_alpha=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.discovered = SymbolicActivation(discovered_expr, label="unbounded_inner")

    def forward(self, x):
        return F.relu(x) + self.alpha * self.discovered(x)

    def extra_repr(self):
        return f"ReLU + {self.alpha.item():.3f}*({self.discovered.formula_str})"


class ManualActivation(nn.Module):
    """Wraps a lambda as a named nn.Module."""
    def __init__(self, func, name):
        super().__init__()
        self._func = func
        self._name = name

    def forward(self, x):
        return self._func(x)

    def extra_repr(self):
        return self._name


MANUAL_CANDIDATES = OrderedDict([
    ("tanh",                 lambda x: torch.tanh(x)),
    ("arctan(x)",            lambda x: torch.atan(x)),
    ("arctan(x+0.5)",       lambda x: torch.atan(x + 0.5)),
    ("arctan(x+0.6)",       lambda x: torch.atan(x + 0.6)),
    ("cos(sqrt|x|)",         lambda x: torch.cos(torch.sqrt(x.abs() + 1e-8))),
    ("cos(sqrt(|x+0.6|))",  lambda x: torch.cos(torch.sqrt((x + 0.6).abs() + 1e-8))),
    ("cos(arctan(x))",      lambda x: torch.cos(torch.atan(x))),
    ("0.64*cos(arctan(x))", lambda x: 0.64 * torch.cos(torch.atan(x))),
    ("1/sqrt(1+x^2)",       lambda x: 1.0 / torch.sqrt(1 + x ** 2)),
    ("x/(1+|x|)",            lambda x: x / (1 + x.abs())),
    ("sin(x)/(1+|x|)",      lambda x: torch.sin(x) / (1 + x.abs())),
])


def get_activation(name, sympy_expr=None, label=""):
    """Factory for all activation types."""
    if name == "ReLU":        return nn.ReLU()
    elif name == "GELU":      return nn.GELU()
    elif name == "SiLU":      return nn.SiLU()
    elif name == "PReLU":     return nn.PReLU()
    elif name == "Mish":      return nn.Mish()
    elif name == "Tanh":      return nn.Tanh()
    elif name == "Sigmoid":   return nn.Sigmoid()
    elif name == "PAU":       return PadeActivationUnit()
    elif name == "Hybrid":    return SymbolicActivation(sympy_expr, label=label)
    elif name == "Unbounded": return UnboundedHybridActivation(sympy_expr)
    elif name.startswith("Manual:"):
        mname = name[7:]
        return ManualActivation(MANUAL_CANDIDATES[mname], mname)
    else:
        raise ValueError(f"Unknown activation: {name}")


STANDARD_ACTIVATIONS = [
    "ReLU", "GELU", "SiLU", "PReLU", "Mish", "Tanh", "Sigmoid", "PAU"
]


# ==========================================================================
# 5. MODEL ARCHITECTURES
# ==========================================================================
class FlexibleModel(nn.Module):
    """Configurable MLP: variable depth, any activation, optional shortcut."""
    def __init__(self, input_dim, act_name, depth=2, sympy_expr=None,
                 use_shortcut=False, label=""):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.shortcut_act = None
        widths = {2: [64, 32], 3: [128, 64, 32], 4: [256, 128, 64, 32]}[depth]
        layers = []
        prev = input_dim
        for w in widths:
            layers += [
                nn.Linear(prev, w), nn.BatchNorm1d(w),
                get_activation(act_name, sympy_expr, label),
                nn.Dropout(0.3),
            ]
            prev = w
        self.backbone = nn.Sequential(*layers)
        head_dim = prev
        if use_shortcut and sympy_expr is not None:
            self.shortcut_act = SymbolicActivation(sympy_expr, label="shortcut")
            head_dim += 1
        self.head = nn.Linear(head_dim, 1)

    def forward(self, x):
        out = self.backbone(x)
        if self.use_shortcut and self.shortcut_act is not None:
            sc = self.shortcut_act(x).mean(dim=1, keepdim=True)
            out = torch.cat([out, sc], dim=1)
        return self.head(out)


class HeavyModel(nn.Module):
    """Large baseline: 200->100->1 with ReLU."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200), nn.BatchNorm1d(200), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(200, 100), nn.BatchNorm1d(100), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Single residual block."""
    def __init__(self, dim, act_module):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
            act_module, nn.Dropout(0.3),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualModel(nn.Module):
    """
    MLP with residual skip connections.
    Addresses bounded-activation gradient collapse at depth.
    """
    def __init__(self, input_dim, act_name, num_blocks=4, hidden_dim=128,
                 sympy_expr=None, label=""):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(
                hidden_dim, get_activation(act_name, sympy_expr, label)))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        return self.head(x)


# ==========================================================================
# 6. TRAINING ENGINE — Early stopping, BCEWithLogitsLoss
# ==========================================================================
def count_params(model, input_dim):
    try:
        import torchinfo
        return torchinfo.summary(
            model, input_data=torch.zeros(2, input_dim).to(DEVICE),
            verbose=0).total_params
    except Exception:
        return sum(p.numel() for p in model.parameters())


def train_and_eval(model, X_train, y_train, X_test, y_test):
    """Train with early stopping on validation split. Returns (acc, auc, f1)."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    best_val, patience_ctr, best_state = float('inf'), 0, None

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(X_val_t), y_val_t).item()
        if vl < best_val:
            best_val, patience_ctr = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        probs = torch.sigmoid(model(xt)).cpu().numpy()
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = 0.5
    f1 = f1_score(y_test, preds, zero_division=0)
    return acc, auc, f1


def run_multi_seed(model_factory, X, y, input_dim, label, seeds=None):
    """Run across all seeds. Returns result dict with raw arrays."""
    seeds = seeds or SEEDS
    accs, aucs, f1s = [], [], []
    params = None
    for seed in seeds:
        set_seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
        model = model_factory()
        if params is None:
            params = count_params(model, input_dim)
        acc, auc, f1 = train_and_eval(model, X_tr, y_tr, X_te, y_te)
        accs.append(acc)
        aucs.append(auc)
        f1s.append(f1)
    return {
        "label": label, "params": params,
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        "accs": accs, "aucs": aucs, "f1s": f1s,
        "eff_mean": np.mean(aucs) / np.log10(max(params, 2)) if params else 0,
    }


# ==========================================================================
# 7. PySR DISCOVERY — Stochasticity + Constrained
# ==========================================================================
def _make_pysr_model(constrained=False, sr_seed=0):
    """Create PySR regressor. If constrained, includes relu operator."""
    unary_ops = [
        "abs", "atan", "exp", "sin", "cos",
        "square(x) = x^2",
        "safe_log(x) = log(abs(x) + convert(typeof(x), 1e-8))",
        "safe_sqrt(x) = sqrt(abs(x) + convert(typeof(x), 1e-8))",
    ]
    extra_mappings = {
        "safe_log": lambda x: sympy.log(sympy.Abs(x) + 1e-8),
        "safe_sqrt": lambda x: sympy.sqrt(sympy.Abs(x) + 1e-8),
        "square": lambda x: x ** 2,
    }
    if constrained:
        unary_ops.append("relu(x) = max(x, zero(x))")
        extra_mappings["relu"] = lambda x: sympy.Max(x, 0)

    return PySRRegressor(
        niterations=PYSR_ITERS,
        populations=PYSR_POPS,
        population_size=PYSR_POP_SIZE,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=unary_ops,
        extra_sympy_mappings=extra_mappings,
        random_state=sr_seed,
        deterministic=True,
        procs=0,
        multithreading=False,
    )


def discover_formula(X_disc, y_disc, dataset_name, label,
                     constrained=False, sr_seed=0):
    """Run PySR and return (sympy_expr, str)."""
    feature_names = [f"x{i}" for i in range(X_disc.shape[1])]
    sr = _make_pysr_model(constrained=constrained, sr_seed=sr_seed)
    sr.fit(X_disc, y_disc, variable_names=feature_names)
    expr = sr.sympy()
    expr_str = str(expr)
    print(f"    [{label}] Best: {expr_str[:80]}")

    # Save Pareto front
    try:
        pf = sr.equations_
        if isinstance(pf, pd.DataFrame):
            pf.to_csv(out_path(f"pareto_{dataset_name}_{label}.csv"), index=False)
    except Exception:
        pass

    return expr, expr_str


def pysr_stochasticity(X_disc, y_disc, dataset_name, n_runs=3):
    """Run PySR n_runs times with different seeds."""
    print(f"\n    [STOCHASTICITY] {dataset_name}: {n_runs} SR runs")
    formulas = []
    for run_idx in range(n_runs):
        sr_seed = 100 + run_idx * 7
        expr, expr_str = discover_formula(
            X_disc, y_disc, dataset_name,
            label=f"stoch_run{run_idx}", sr_seed=sr_seed)
        formulas.append({
            "run": run_idx, "seed": sr_seed,
            "formula": expr_str, "sympy": expr,
        })
    return formulas


# ==========================================================================
# 8. ANALYSIS FUNCTIONS
# ==========================================================================

# --- Abstraction Analysis ---
def abstraction_analysis(sympy_expr, X_sample, dataset_name):
    """Compute Pearson r between full multi-var and abstracted univariate."""
    symbols_in_expr = list(sympy_expr.free_symbols)
    if not symbols_in_expr:
        print(f"    [{dataset_name}] Constant expression. r=1.0 (trivial)")
        return 1.0

    x_sym = sympy.Symbol('x')
    generalized = sympy_expr
    for s in symbols_in_expr:
        generalized = generalized.subs(s, x_sym)

    numpy_map = {
        "safe_sqrt": lambda t: np.sqrt(np.abs(t) + 1e-8),
        "square": lambda t: t ** 2,
        "relu": lambda t: np.maximum(t, 0),
    }

    try:
        full_func = sympy.lambdify(
            sorted(symbols_in_expr, key=str),
            sympy_expr, modules=[numpy_map, "numpy"])
        gen_func = sympy.lambdify(
            x_sym, generalized, modules=[numpy_map, "numpy"])

        X_sub = X_sample[:1000]
        sym_to_col = {}
        for s in sorted(symbols_in_expr, key=str):
            m = re.search(r'(\d+)', str(s))
            if m:
                idx = int(m.group(1))
                if idx < X_sub.shape[1]:
                    sym_to_col[s] = idx
        if len(sym_to_col) != len(symbols_in_expr):
            print(f"    [{dataset_name}] Could not map symbols -> columns.")
            return float('nan')

        y_full = np.array(full_func(
            *[X_sub[:, sym_to_col[s]]
              for s in sorted(symbols_in_expr, key=str)]
        ), dtype=float).flatten()

        y_gen = np.mean(
            [gen_func(X_sub[:, c])
             for c in range(min(X_sub.shape[1], 10))],
            axis=0).flatten()

        ml = min(len(y_full), len(y_gen))
        y_full, y_gen = y_full[:ml], y_gen[:ml]
        mask = np.isfinite(y_full) & np.isfinite(y_gen)
        y_full, y_gen = y_full[mask], y_gen[mask]

        if len(y_full) < 10:
            return float('nan')

        r = np.corrcoef(y_full, y_gen)[0, 1]
        print(f"    [{dataset_name}] Abstraction Pearson r = {r:.4f}")
        return r

    except Exception as e:
        print(f"    [{dataset_name}] Abstraction analysis failed: {e}")
        return float('nan')


# --- Quantitative Spectral Analysis ---
def spectral_analysis_full(activations_dict, save_prefix="spectral"):
    """Full spectral analysis with quantitative similarity matrices."""
    x = torch.linspace(-5, 5, 1024)
    N = len(x)
    dx = (x[-1] - x[0]).item() / N
    freqs = fftfreq(N, d=dx)[:N // 2]

    baselines = OrderedDict([
        ("ReLU", torch.relu(x).numpy()),
        ("GELU", F.gelu(x).numpy()),
        ("SiLU", F.silu(x).numpy()),
        ("Mish", F.mish(x).numpy()),
        ("Tanh", torch.tanh(x).numpy()),
    ])

    all_spectra = OrderedDict()
    all_shapes = OrderedDict()

    for name, y in baselines.items():
        all_spectra[name] = np.abs(fft(y))[:N // 2]
        all_shapes[name] = y

    for name, act in activations_dict.items():
        try:
            with torch.no_grad():
                y = act(x).numpy()
            if np.all(np.isfinite(y)):
                all_spectra[name] = np.abs(fft(y))[:N // 2]
                all_shapes[name] = y
        except Exception:
            pass

    names = list(all_spectra.keys())
    n = len(names)
    cos_sim = np.zeros((n, n))
    kl_div = np.zeros((n, n))

    for i in range(n):
        si = all_spectra[names[i]] + 1e-12
        si_norm = si / si.sum()
        for j in range(n):
            sj = all_spectra[names[j]] + 1e-12
            sj_norm = sj / sj.sum()
            cos_sim[i, j] = (
                np.dot(si, sj)
                / (np.linalg.norm(si) * np.linalg.norm(sj) + 1e-12)
            )
            kl_div[i, j] = 0.5 * (
                np.sum(si_norm * np.log(si_norm / sj_norm))
                + np.sum(sj_norm * np.log(sj_norm / si_norm))
            )

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    ax1, ax2, ax3 = axes

    cmap_bl = plt.cm.Greys(np.linspace(0.3, 0.7, len(baselines)))
    for i, (name, y) in enumerate(baselines.items()):
        ax1.plot(x.numpy(), y, '--', color=cmap_bl[i], lw=1, alpha=0.6,
                 label=name)
        ax2.plot(freqs[:60], all_spectra[name][:60], '--', color=cmap_bl[i],
                 lw=1, alpha=0.5)

    disc_names = [n for n in activations_dict.keys()]
    cmap_d = plt.cm.tab10(np.linspace(0, 0.8, max(len(disc_names), 1)))
    for i, name in enumerate(disc_names):
        if name in all_shapes:
            ax1.plot(x.numpy(), all_shapes[name], '-', color=cmap_d[i],
                     lw=2, label=name)
            ax2.plot(freqs[:60], all_spectra[name][:60], '-', color=cmap_d[i],
                     lw=2, label=name)

    ax1.set_title("Activation Shapes")
    ax1.legend(fontsize=6)
    ax1.grid(alpha=0.3)
    ax2.set_title("|FFT| Spectra")
    ax2.legend(fontsize=6)
    ax2.grid(alpha=0.3)

    im = ax3.imshow(cos_sim, cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    ax3.set_yticklabels(names, fontsize=6)
    ax3.set_title("Spectral Cosine Similarity")
    for i in range(n):
        for j in range(n):
            ax3.text(j, i, f"{cos_sim[i, j]:.2f}", ha='center', va='center',
                     fontsize=5)
    fig.colorbar(im, ax=ax3, shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path(f"{save_prefix}_full.png"), dpi=300,
                bbox_inches='tight')
    plt.savefig(out_path(f"{save_prefix}_full.pdf"), bbox_inches='tight')
    plt.close()

    pd.DataFrame(cos_sim, index=names, columns=names).to_csv(
        out_path(f"{save_prefix}_cosine.csv"))
    pd.DataFrame(kl_div, index=names, columns=names).to_csv(
        out_path(f"{save_prefix}_kl.csv"))

    print(f"  Saved: {save_prefix}_full.png/.pdf, "
          f"{save_prefix}_cosine.csv, {save_prefix}_kl.csv")
    return names, cos_sim, kl_div


# --- Full Transfer Matrix ---
def compute_transfer_matrix(discovered_exprs, datasets_cache,
                            transfer_datasets):
    """Compute AUC for every (source, target) pair."""
    n = len(transfer_datasets)
    matrix_auc = np.full((n, n), np.nan)
    matrix_std = np.full((n, n), np.nan)
    detail_rows = []

    for i, src in enumerate(transfer_datasets):
        if src not in discovered_exprs:
            continue
        expr = discovered_exprs[src]
        for j, tgt in enumerate(transfer_datasets):
            if tgt not in datasets_cache:
                continue
            X, y = datasets_cache[tgt]
            input_dim = X.shape[1]
            label = f"Transfer({src}->{tgt})"
            print(f"    {label}...")
            r = run_multi_seed(
                lambda e=expr, d=input_dim: FlexibleModel(
                    d, "Hybrid", depth=2, sympy_expr=e),
                X, y, input_dim, label)
            matrix_auc[i, j] = r["auc_mean"]
            matrix_std[i, j] = r["auc_std"]
            detail_rows.append({
                "Source": src, "Target": tgt,
                "AUC_mean": r["auc_mean"], "AUC_std": r["auc_std"],
                "Is_specialist": (src == tgt),
            })

    return matrix_auc, matrix_std, detail_rows


def plot_transfer_heatmap(matrix_auc, dataset_names,
                          save_name="transfer_heatmap"):
    """Plot NxN transfer heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    n = len(dataset_names)
    mask = np.isnan(matrix_auc)
    plot_data = np.where(mask, 0.5, matrix_auc)

    im = ax.imshow(plot_data, cmap='YlOrRd', aspect='equal',
                   vmin=0.5, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(dataset_names, fontsize=8)
    ax.set_xlabel("Target Dataset", fontsize=12)
    ax.set_ylabel("Source Dataset (activation from)", fontsize=12)
    ax.set_title("Transfer Matrix: AUC-ROC", fontsize=14, fontweight='bold')

    for i in range(n):
        for j in range(n):
            v = matrix_auc[i, j]
            if not np.isnan(v):
                color = 'white' if v < 0.7 else 'black'
                weight = 'bold' if i == j else 'normal'
                ax.text(j, i, f"{v:.3f}", ha='center', va='center',
                        fontsize=7, color=color, fontweight=weight)
            else:
                ax.text(j, i, "N/A", ha='center', va='center',
                        fontsize=7, color='gray')

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path(f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(out_path(f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_name}.png/.pdf")


# --- Statistical Significance ---
def significance_tests(results_by_dataset):
    """Wilcoxon signed-rank: Hybrid vs each baseline, per dataset."""
    sig_rows = []
    for dname, results in results_by_dataset.items():
        hybrids = [r for r in results
                   if "Hybrid" in r["label"] or "Unbounded" in r["label"]]
        baselines = [r for r in results if r not in hybrids]
        for h in hybrids:
            for b in baselines:
                ha, ba = np.array(h["aucs"]), np.array(b["aucs"])
                if len(ha) >= 6 and len(ba) >= 6:
                    try:
                        _, p = wilcoxon(ha, ba, alternative='two-sided')
                    except ValueError:
                        p = 1.0
                else:
                    p = float('nan')
                sig_rows.append({
                    "Dataset": dname, "Hybrid": h["label"],
                    "Baseline": b["label"], "p_value": p,
                    "Hybrid_AUC": np.mean(ha),
                    "Baseline_AUC": np.mean(ba),
                })
    return sig_rows


# --- Manual Search Ablation ---
def manual_search_ablation(X, y, input_dim, dataset_name):
    """Grid search over human-intuitive activations."""
    print(f"\n  MANUAL SEARCH ABLATION: {dataset_name}")
    results = []
    for mname in MANUAL_CANDIDATES:
        label = f"Manual:{mname}"
        print(f"    {label}...")
        r = run_multi_seed(
            lambda mn=mname, d=input_dim: FlexibleModel(
                d, f"Manual:{mn}", depth=2),
            X, y, input_dim, label,
            seeds=SEEDS[:N_MANUAL_SEEDS])
        results.append(r)
        print(f"      AUC={r['auc_mean']:.4f}+/-{r['auc_std']:.4f}")
    return results


# ==========================================================================
# 9. PLOTTING UTILITIES
# ==========================================================================
def plot_efficiency_frontier(all_results, save_name="efficiency_frontier"):
    fig, ax = plt.subplots(figsize=(14, 8))
    markers_map = {
        "HIGGS": "o", "FOREST_COVER": "s", "SPAMBASE": "^",
        "MINIBOONE": "D", "MAGIC_TELESCOPE": "P",
        "PHONEME": "X", "EEG_EYE_STATE": "v", "WILT": "<",
    }
    cmap = plt.cm.tab10
    legend_handles = []
    for di, (dname, results) in enumerate(all_results.items()):
        mk = markers_map.get(dname, "o")
        color = cmap(di % 10)
        for r in results:
            is_h = "Hybrid" in r["label"] or "Unbounded" in r["label"]
            ax.errorbar(
                np.log10(max(r["params"], 2)), r["auc_mean"],
                yerr=r["auc_std"],
                marker=mk, color='red' if is_h else color,
                markersize=8 if is_h else 5, capsize=2, alpha=0.8,
                linewidth=0,
                markeredgecolor='black' if is_h else 'gray',
            )
        legend_handles.append(
            plt.Line2D([0], [0], marker=mk, color=color, linestyle='',
                       label=dname, markersize=6))

    legend_handles.append(
        plt.Line2D([0], [0], marker='o', color='red', linestyle='',
                   label='Hybrid/Unbounded', markersize=8,
                   markeredgecolor='black'))
    ax.legend(handles=legend_handles, fontsize=7, loc='lower right')
    ax.set_xlabel("log10(Parameters)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Efficiency Frontier ({len(SEEDS)} seeds)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path(f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(out_path(f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()


def plot_fraction_sweep(sweep_results, best_formulas,
                        save_name="fraction_sweep"):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for di, (dname, frac_results) in enumerate(sweep_results.items()):
        fracs = [r["fraction"] for r in frac_results]
        aucs = [r["auc_mean"] for r in frac_results]
        stds = [r["auc_std"] for r in frac_results]
        ax.errorbar(fracs, aucs, yerr=stds, marker='o', color=cmap(di),
                     linewidth=2, capsize=4, label=dname)
        if dname in best_formulas:
            bf = best_formulas[dname]
            ax.plot(bf["fraction"], bf["auc_mean"], '*', color=cmap(di),
                    markersize=18, markeredgecolor='black', zorder=5)
    ax.set_xlabel("Discovery Data Fraction")
    ax.set_ylabel("AUC-ROC (Hybrid)")
    ax.set_xticks(DATA_FRACTIONS)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in DATA_FRACTIONS])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path(f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(out_path(f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()


def plot_stochasticity(stoch_results, save_name="sr_stochasticity"):
    """Plot PySR stochasticity: activation shapes across runs per dataset."""
    n_plots = len(stoch_results)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    x = torch.linspace(-5, 5, 512)
    for ax, (dname, runs) in zip(axes, stoch_results.items()):
        ax.set_title(f"{dname}: {len(runs)} SR runs", fontweight='bold')
        for ri, run in enumerate(runs):
            try:
                act = SymbolicActivation(run["sympy"],
                                          label=f"run{ri}")
                with torch.no_grad():
                    y = act(x).numpy()
                if np.all(np.isfinite(y)):
                    ax.plot(x.numpy(), y, '-', alpha=0.7,
                            label=f"Run {ri}: {run['formula'][:30]}...")
            except Exception:
                pass
        ax.legend(fontsize=5)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path(f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(out_path(f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()


def plot_manual_vs_sr(manual_results, sr_result, dataset_name,
                      save_name=None):
    """Bar chart: manual candidates vs SR-discovered activation."""
    labels = [r["label"].replace("Manual:", "") for r in manual_results]
    aucs = [r["auc_mean"] for r in manual_results]
    stds = [r["auc_std"] for r in manual_results]

    labels.append(f"SR ({sr_result['label']})")
    aucs.append(sr_result["auc_mean"])
    stds.append(sr_result["auc_std"])

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['steelblue'] * len(manual_results) + ['red']
    ax.barh(range(len(labels)), aucs, xerr=stds, color=colors, alpha=0.8,
            capsize=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC-ROC")
    ax.set_title(f"Manual Search vs SR: {dataset_name}")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fname = save_name or f"manual_vs_sr_{dataset_name}.png"
    plt.savefig(out_path(fname), dpi=300, bbox_inches='tight')
    plt.close()


def plot_depth_ablation(depth_results, save_name="depth_ablation"):
    """Bar chart per dataset: depth x activation type."""
    n_datasets = len(depth_results)
    if n_datasets == 0:
        return
    fig, axes = plt.subplots(1, n_datasets,
                              figsize=(7 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]
    for ax, (dname, results) in zip(axes, depth_results.items()):
        labels = [r["label"] for r in results]
        aucs = [r["auc_mean"] for r in results]
        stds = [r["auc_std"] for r in results]
        colors = ['steelblue' if 'ReLU' in l else 'red' if 'Hybrid' in l
                  else 'green' for l in labels]
        ax.barh(range(len(labels)), aucs, xerr=stds, color=colors,
                alpha=0.8, capsize=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("AUC-ROC")
        ax.set_title(f"Depth Ablation: {dname}", fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path(f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(out_path(f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()


def plot_component_ablation(ablation_results,
                            save_name="component_ablation"):
    """Bar chart per dataset: component ablation."""
    n_datasets = len(ablation_results)
    if n_datasets == 0:
        return
    fig, axes = plt.subplots(1, n_datasets,
                              figsize=(7 * n_datasets, 4))
    if n_datasets == 1:
        axes = [axes]
    for ax, (dname, results) in zip(axes, ablation_results.items()):
        labels = [r["label"] for r in results]
        aucs = [r["auc_mean"] for r in results]
        stds = [r["auc_std"] for r in results]
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        ax.barh(range(len(labels)), aucs, xerr=stds, color=colors,
                alpha=0.8, capsize=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("AUC-ROC")
        ax.set_title(f"Component Ablation: {dname}", fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path(f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(out_path(f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()


def plot_main_benchmark_bars(all_results, save_name="main_benchmark"):
    """Grouped bar chart of main benchmark results per dataset."""
    n_datasets = len(all_results)
    if n_datasets == 0:
        return
    cols = min(n_datasets, 4)
    rows = (n_datasets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx, (dname, results) in enumerate(all_results.items()):
        ax = axes[idx]
        labels = [r["label"] for r in results]
        aucs = [r["auc_mean"] for r in results]
        stds = [r["auc_std"] for r in results]
        colors = []
        for l in labels:
            if "Heavy" in l:
                colors.append('gray')
            elif "Hybrid" in l or "Unbounded" in l:
                colors.append('red')
            elif "Constrained" in l:
                colors.append('orange')
            else:
                colors.append('steelblue')
        ax.barh(range(len(labels)), aucs, xerr=stds, color=colors,
                alpha=0.8, capsize=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("AUC-ROC")
        ax.set_title(dname, fontweight='bold', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path(f"{save_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(out_path(f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()


# ==========================================================================
# 10. LOGGING UTILITY
# ==========================================================================
class Logger:
    """Tee stdout to both console and a log file inside the output dir."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ==========================================================================
# 11. MAIN EXECUTION
# ==========================================================================
def main():
    # Start logging
    logger = Logger(out_path("benchmark_log.txt"))
    sys.stdout = logger

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Timestamp: {TIMESTAMP}\n")

    all_results = OrderedDict()
    discovered_exprs = {}
    discovered_all_fracs = {}
    sweep_results = {}
    best_formulas = {}
    stochasticity_results = {}
    constrained_exprs = {}
    abstraction_corrs = {}
    dataset_cache = {}

    # ==================================================================
    # PHASE 1: LOAD ALL DATASETS
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 1: LOADING {len(DATASETS_CONFIG)} DATASETS")
    print(f"{'=' * 70}")

    for dname, cfg in DATASETS_CONFIG.items():
        try:
            set_seed(42)
            X, y = get_dataset(dname, limit_rows=cfg["limit"])
            dataset_cache[dname] = (X, y)
            print(f"  OK {dname}: shape={X.shape}, pos_rate={y.mean():.3f}, "
                  f"domain={cfg['domain']}, geometry={cfg['geometry']}")
        except Exception as e:
            print(f"  FAIL {dname}: {e}")

    # Save dataset summary
    ds_summary = []
    for dname in dataset_cache:
        X, y = dataset_cache[dname]
        cfg = DATASETS_CONFIG[dname]
        ds_summary.append({
            "Dataset": dname, "n_samples": len(y),
            "n_features": X.shape[1],
            "pos_rate": float(y.mean()),
            "domain": cfg["domain"],
            "geometry": cfg["geometry"],
        })
    pd.DataFrame(ds_summary).to_csv(
        out_path("dataset_summary.csv"), index=False)

    # ==================================================================
    # PHASE 2: PySR DISCOVERY
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: PySR DISCOVERY")
    print(f"{'=' * 70}")

    if not HAS_JULIA:
        print("  [SKIP] PySR not available. Loading fallback formulas.")
        FALLBACK = {
            "HIGGS": "cos(atan(x25)) * 0.638",
            "FOREST_COVER": "cos(safe_sqrt(-0.632 - x0))",
            "SPAMBASE": "atan(x52 + 0.557)",
        }
        for dname, fstr in FALLBACK.items():
            if dname in dataset_cache:
                expr = parse_formula(fstr)
                discovered_exprs[dname] = expr
                discovered_all_fracs[dname] = {1.0: (expr, fstr)}
                best_formulas[dname] = {
                    "fraction": 1.0, "frac_label": "100pct",
                    "formula_str": fstr, "expr": expr,
                    "auc_mean": 0.0, "auc_std": 0.0,
                }
    else:
        # --- Fraction sweep for original 3 datasets ---
        SWEEP_DATASETS = ["HIGGS", "FOREST_COVER", "SPAMBASE"]

        for dname in SWEEP_DATASETS:
            if dname not in dataset_cache:
                continue
            X, y = dataset_cache[dname]
            input_dim = X.shape[1]

            n_pool = min(DISCOVERY_ROWS, int(0.3 * len(X)))
            set_seed(42)
            idx_pool = np.random.choice(len(X), n_pool, replace=False)
            X_pool = StandardScaler().fit_transform(X[idx_pool])
            y_pool = y[idx_pool]

            discovered_all_fracs[dname] = {}
            frac_results_list = []

            print(f"\n  [{dname}] Fraction sweep:")
            for frac in DATA_FRACTIONS:
                n_use = max(100, int(frac * len(X_pool)))
                fl = FRAC_LABEL(frac)
                X_disc, y_disc = X_pool[:n_use], y_pool[:n_use]

                t0 = time.time()
                expr, expr_str = discover_formula(
                    X_disc, y_disc, dname, fl)
                elapsed = time.time() - t0
                print(f"    {fl} ({n_use} samples, {elapsed:.0f}s): "
                      f"{expr_str[:60]}")

                discovered_all_fracs[dname][frac] = (expr, expr_str)

                # Multi-seed eval for proper fraction selection
                mask_nn = np.ones(len(X), dtype=bool)
                mask_nn[idx_pool[:n_pool]] = False
                X_nn, y_nn = X[mask_nn], y[mask_nn]

                r = run_multi_seed(
                    lambda e=expr, d=input_dim: FlexibleModel(
                        d, "Hybrid", depth=2, sympy_expr=e),
                    X_nn, y_nn, input_dim, f"Hybrid_{fl}")
                r.update({
                    "fraction": frac, "frac_label": fl,
                    "formula_str": expr_str,
                })
                frac_results_list.append(r)
                print(f"      AUC={r['auc_mean']:.4f}+/-{r['auc_std']:.4f}")

            sweep_results[dname] = frac_results_list

            # Select best fraction by highest mean AUC
            best_r = max(frac_results_list, key=lambda r: r["auc_mean"])
            best_frac = best_r["fraction"]
            best_formulas[dname] = {
                "fraction": best_frac,
                "frac_label": best_r["frac_label"],
                "formula_str": best_r["formula_str"],
                "expr": discovered_all_fracs[dname][best_frac][0],
                "auc_mean": best_r["auc_mean"],
                "auc_std": best_r["auc_std"],
            }
            discovered_exprs[dname] = best_formulas[dname]["expr"]
            print(f"    -> BEST fraction: {best_r['frac_label']} "
                  f"(AUC={best_r['auc_mean']:.4f})")

        # --- Single 100% run for new datasets ---
        NEW_DATASETS = [d for d in DATASETS_CONFIG if d not in SWEEP_DATASETS]
        for dname in NEW_DATASETS:
            if dname not in dataset_cache:
                continue
            X, y = dataset_cache[dname]
            n_pool = min(DISCOVERY_ROWS, int(0.3 * len(X)))
            set_seed(42)
            idx = np.random.choice(len(X), n_pool, replace=False)
            X_pool = StandardScaler().fit_transform(X[idx])
            y_pool = y[idx]

            print(f"\n  [{dname}] Single SR run (100%):")
            expr, expr_str = discover_formula(
                X_pool, y_pool, dname, "100pct")
            discovered_exprs[dname] = expr
            discovered_all_fracs[dname] = {1.0: (expr, expr_str)}
            best_formulas[dname] = {
                "fraction": 1.0, "frac_label": "100pct",
                "formula_str": expr_str, "expr": expr,
                "auc_mean": 0.0, "auc_std": 0.0,
            }

        # --- Stochasticity characterization ---
        print(f"\n{'=' * 70}")
        print(f"  PySR STOCHASTICITY")
        print(f"{'=' * 70}")
        for dname in SWEEP_DATASETS:
            if dname not in dataset_cache:
                continue
            X, y = dataset_cache[dname]
            n_pool = min(DISCOVERY_ROWS, int(0.3 * len(X)))
            set_seed(42)
            idx = np.random.choice(len(X), n_pool, replace=False)
            X_pool = StandardScaler().fit_transform(X[idx])
            y_pool = y[idx]
            stochasticity_results[dname] = pysr_stochasticity(
                X_pool, y_pool, dname, n_runs=N_SR_RUNS)

        # --- Constrained SR ---
        print(f"\n{'=' * 70}")
        print(f"  CONSTRAINED SR (relu operator)")
        print(f"{'=' * 70}")
        for dname in ["HIGGS", "FOREST_COVER"]:
            if dname not in dataset_cache:
                continue
            X, y = dataset_cache[dname]
            n_pool = min(DISCOVERY_ROWS, int(0.3 * len(X)))
            set_seed(42)
            idx = np.random.choice(len(X), n_pool, replace=False)
            X_pool = StandardScaler().fit_transform(X[idx])
            y_pool = y[idx]
            expr, expr_str = discover_formula(
                X_pool, y_pool, dname, "constrained", constrained=True)
            constrained_exprs[dname] = (expr, expr_str)
            print(f"    [{dname}] Constrained: {expr_str[:60]}")

    # Save fraction sweep
    if sweep_results:
        rows = [{"Dataset": d, **{k: v for k, v in r.items()
                                   if k not in ('accs', 'aucs', 'f1s')}}
                for d, frs in sweep_results.items() for r in frs]
        pd.DataFrame(rows).to_csv(
            out_path("fraction_sweep_results.csv"), index=False)
        plot_fraction_sweep(sweep_results, best_formulas)

    # Save stochasticity
    if stochasticity_results:
        stoch_rows = []
        for dname, runs in stochasticity_results.items():
            for run in runs:
                stoch_rows.append({
                    "Dataset": dname, "Run": run["run"],
                    "Seed": run["seed"], "Formula": run["formula"],
                })
        pd.DataFrame(stoch_rows).to_csv(
            out_path("sr_stochasticity.csv"), index=False)
        plot_stochasticity(stochasticity_results)

    # Save constrained SR formulas
    if constrained_exprs:
        c_rows = [{"Dataset": d, "Formula": s}
                   for d, (_, s) in constrained_exprs.items()]
        pd.DataFrame(c_rows).to_csv(
            out_path("constrained_sr_formulas.csv"), index=False)

    # ==================================================================
    # PHASE 3: ANALYSIS
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 3: ANALYSIS")
    print(f"{'=' * 70}")

    # --- Abstraction for ALL datasets ---
    print(f"\n  Abstraction Analysis (all datasets):")
    for dname, expr in discovered_exprs.items():
        if dname in dataset_cache:
            X, y = dataset_cache[dname]
            X_sc = StandardScaler().fit_transform(X[:5000])
            r = abstraction_analysis(expr, X_sc, dname)
            abstraction_corrs[dname] = r

    pd.DataFrame([{"Dataset": d, "Pearson_r": r}
                   for d, r in abstraction_corrs.items()]).to_csv(
        out_path("abstraction_correlations.csv"), index=False)

    # --- Spectral analysis ---
    spectral_acts = OrderedDict()
    for dname, expr in discovered_exprs.items():
        try:
            spectral_acts[f"Disc({dname})"] = SymbolicActivation(
                expr, label=dname)
        except Exception:
            pass
    if spectral_acts:
        spectral_analysis_full(spectral_acts, save_prefix="spectral")

    # ==================================================================
    # PHASE 4: MAIN BENCHMARK
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 4: MAIN BENCHMARK")
    print(f"  {len(STANDARD_ACTIVATIONS)} standard + Specialist + Unbounded")
    print(f"  x {len(dataset_cache)} datasets x {len(SEEDS)} seeds "
          f"x {EPOCHS} epochs")
    print(f"{'=' * 70}")

    for dname in dataset_cache:
        X, y = dataset_cache[dname]
        input_dim = X.shape[1]
        expr = discovered_exprs.get(dname)
        results = []

        print(f"\n  {'=' * 50}")
        print(f"  {dname} (n={len(X)}, d={input_dim})")
        print(f"  {'=' * 50}")

        # Heavy baseline
        print(f"    Heavy (ReLU)...")
        results.append(run_multi_seed(
            lambda d=input_dim: HeavyModel(d),
            X, y, input_dim, "Heavy (ReLU)"))

        # Standard activations on light MLP
        for act_name in STANDARD_ACTIVATIONS:
            print(f"    Light {act_name}...")
            results.append(run_multi_seed(
                lambda an=act_name, d=input_dim: FlexibleModel(
                    d, an, depth=2),
                X, y, input_dim, act_name))

        # Specialist
        if expr is not None:
            print(f"    Hybrid (Specialist)...")
            results.append(run_multi_seed(
                lambda e=expr, d=input_dim: FlexibleModel(
                    d, "Hybrid", depth=2, sympy_expr=e),
                X, y, input_dim, "Hybrid (Specialist)"))

            # Unbounded Hybrid
            print(f"    Unbounded Hybrid...")
            results.append(run_multi_seed(
                lambda e=expr, d=input_dim: FlexibleModel(
                    d, "Unbounded", depth=2, sympy_expr=e),
                X, y, input_dim, "Unbounded Hybrid"))

            # Constrained SR
            if dname in constrained_exprs:
                c_expr, _ = constrained_exprs[dname]
                print(f"    Hybrid (Constrained SR)...")
                results.append(run_multi_seed(
                    lambda e=c_expr, d=input_dim: FlexibleModel(
                        d, "Hybrid", depth=2, sympy_expr=e),
                    X, y, input_dim, "Hybrid (Constrained)"))

        # Print summary
        for r in results:
            print(f"    {r['label']:<30} "
                  f"AUC={r['auc_mean']:.4f}+/-{r['auc_std']:.4f} "
                  f"Params={r['params']} Eff={r['eff_mean']:.4f}")

        all_results[dname] = results

    # ==================================================================
    # PHASE 5: TRANSFER MATRIX
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 5: TRANSFER MATRIX")
    print(f"{'=' * 70}")

    transfer_datasets = [
        d for d in TRANSFER_DATASETS
        if d in discovered_exprs and d in dataset_cache
    ]
    print(f"  Transfer datasets: {transfer_datasets}")
    print(f"  Matrix size: {len(transfer_datasets)}x{len(transfer_datasets)}")

    transfer_detail_rows = []
    if len(transfer_datasets) >= 2:
        t_matrix, t_std, t_rows = compute_transfer_matrix(
            discovered_exprs, dataset_cache, transfer_datasets)
        transfer_detail_rows = t_rows
        pd.DataFrame(t_rows).to_csv(
            out_path("transfer_matrix.csv"), index=False)
        plot_transfer_heatmap(t_matrix, transfer_datasets)

        # Print matrix
        print(f"\n  Transfer Matrix (AUC):")
        header = f"  {'':>18}"
        for d in transfer_datasets:
            header += f" {d[:12]:>14}"
        print(header)
        for i, src in enumerate(transfer_datasets):
            row_str = f"  {src:>18}"
            for j in range(len(transfer_datasets)):
                v = t_matrix[i, j]
                marker = " *" if i == j else "  "
                if np.isnan(v):
                    row_str += f" {'N/A':>12}{marker}"
                else:
                    row_str += f" {v:>12.4f}{marker}"
            print(row_str)

    # ==================================================================
    # PHASE 6: DEPTH ABLATION
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 6: DEPTH ABLATION")
    print(f"{'=' * 70}")

    depth_results = OrderedDict()
    for dname in DEPTH_DATASETS:
        if dname not in dataset_cache or dname not in discovered_exprs:
            continue
        X, y = dataset_cache[dname]
        input_dim = X.shape[1]
        expr = discovered_exprs[dname]
        d_results = []

        print(f"\n  {dname}:")

        # Standard depth sweep
        for depth in [2, 3, 4]:
            for act_name in ["ReLU", "Hybrid"]:
                label = f"Depth-{depth} {act_name}"
                print(f"    {label}...")
                e = expr if act_name == "Hybrid" else None
                d_results.append(run_multi_seed(
                    lambda d=depth, a=act_name, ee=e, dim=input_dim:
                        FlexibleModel(dim, a, depth=d, sympy_expr=ee),
                    X, y, input_dim, label))

        # Residual MLP
        for num_blocks in [4, 8]:
            for act_name in ["ReLU", "Hybrid"]:
                label = f"Residual-{num_blocks}b {act_name}"
                print(f"    {label}...")
                e = expr if act_name == "Hybrid" else None
                d_results.append(run_multi_seed(
                    lambda a=act_name, ee=e, nb=num_blocks, dim=input_dim:
                        ResidualModel(dim, a, num_blocks=nb, sympy_expr=ee),
                    X, y, input_dim, label))

        depth_results[dname] = d_results

        for r in d_results:
            print(f"    {r['label']:<30} "
                  f"AUC={r['auc_mean']:.4f}+/-{r['auc_std']:.4f} "
                  f"Params={r['params']}")

    plot_depth_ablation(depth_results)

    # ==================================================================
    # PHASE 7: COMPONENT ABLATION
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 7: COMPONENT ABLATION")
    print(f"{'=' * 70}")

    ablation_results = OrderedDict()
    for dname in ABLATION_DATASETS:
        if dname not in dataset_cache or dname not in discovered_exprs:
            continue
        X, y = dataset_cache[dname]
        input_dim = X.shape[1]
        expr = discovered_exprs[dname]
        a_results = []

        print(f"\n  {dname}:")
        configs = [
            ("ReLU baseline",   "ReLU",   None,  False),
            ("Activation-only", "Hybrid", expr,  False),
            ("Shortcut-only",   "ReLU",   expr,  True),
            ("Full Hybrid",     "Hybrid", expr,  True),
        ]
        for label, act, e, sc in configs:
            print(f"    {label}...")
            a_results.append(run_multi_seed(
                lambda a=act, ee=e, s=sc, dim=input_dim:
                    FlexibleModel(dim, a, depth=2, sympy_expr=ee,
                                  use_shortcut=s),
                X, y, input_dim, label))

        ablation_results[dname] = a_results

    plot_component_ablation(ablation_results)

    # ==================================================================
    # PHASE 8: MANUAL SEARCH ABLATION
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 8: MANUAL SEARCH ABLATION")
    print(f"{'=' * 70}")

    manual_results_all = OrderedDict()
    for dname in dataset_cache:
        X, y = dataset_cache[dname]
        input_dim = X.shape[1]
        manual_res = manual_search_ablation(X, y, input_dim, dname)
        manual_results_all[dname] = manual_res

        # Plot manual vs SR
        if dname in all_results:
            sr_result = next(
                (r for r in all_results[dname]
                 if "Specialist" in r["label"]), None)
            if sr_result:
                plot_manual_vs_sr(manual_res, sr_result, dname)

    # ==================================================================
    # PHASE 9: STATISTICAL SIGNIFICANCE
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 9: STATISTICAL SIGNIFICANCE")
    print(f"{'=' * 70}")

    sig_rows = significance_tests(all_results)

    for dname in all_results:
        dname_sigs = [s for s in sig_rows if s["Dataset"] == dname]
        if dname_sigs:
            print(f"\n  {dname}:")
            print(f"    {'Hybrid':<25} {'vs Baseline':<25} "
                  f"{'p-value':<10} {'H_AUC':<8} {'B_AUC'}")
            for s in dname_sigs:
                sig_marker = (
                    " ***" if s["p_value"] < 0.001 else
                    " **" if s["p_value"] < 0.01 else
                    " *" if s["p_value"] < 0.05 else "")
                print(f"    {s['Hybrid']:<25} {s['Baseline']:<25} "
                      f"{s['p_value']:<10.4f} "
                      f"{s['Hybrid_AUC']:<8.4f} "
                      f"{s['Baseline_AUC']:.4f}{sig_marker}")

    # ==================================================================
    # PHASE 10: SAVE ALL OUTPUTS
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 10: SAVING OUTPUTS")
    print(f"{'=' * 70}")

    # --- Main results CSV ---
    all_rows = []
    for dname, results in all_results.items():
        for r in results:
            all_rows.append({
                "Dataset": dname, "Model": r["label"],
                "Experiment": "Main",
                "AUC_mean": r["auc_mean"], "AUC_std": r["auc_std"],
                "Acc_mean": r["acc_mean"], "Acc_std": r["acc_std"],
                "F1_mean": r["f1_mean"], "F1_std": r["f1_std"],
                "Params": r["params"], "Efficiency": r["eff_mean"],
            })
    for dname, results in depth_results.items():
        for r in results:
            all_rows.append({
                "Dataset": dname, "Model": r["label"],
                "Experiment": "Depth",
                "AUC_mean": r["auc_mean"], "AUC_std": r["auc_std"],
                "Acc_mean": r["acc_mean"], "Acc_std": r["acc_std"],
                "F1_mean": r["f1_mean"], "F1_std": r["f1_std"],
                "Params": r["params"],
            })
    for dname, results in ablation_results.items():
        for r in results:
            all_rows.append({
                "Dataset": dname, "Model": r["label"],
                "Experiment": "Ablation",
                "AUC_mean": r["auc_mean"], "AUC_std": r["auc_std"],
                "Acc_mean": r["acc_mean"], "Acc_std": r["acc_std"],
                "F1_mean": r["f1_mean"], "F1_std": r["f1_std"],
                "Params": r["params"],
            })
    for dname, results in manual_results_all.items():
        for r in results:
            all_rows.append({
                "Dataset": dname, "Model": r["label"],
                "Experiment": "Manual",
                "AUC_mean": r["auc_mean"], "AUC_std": r["auc_std"],
                "Acc_mean": r["acc_mean"], "Acc_std": r["acc_std"],
                "F1_mean": r["f1_mean"], "F1_std": r["f1_std"],
                "Params": r["params"],
            })

    pd.DataFrame(all_rows).to_csv(
        out_path("nsad_bench_results.csv"), index=False)

    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(
            out_path("significance_tests.csv"), index=False)

    if transfer_detail_rows:
        pd.DataFrame(transfer_detail_rows).to_csv(
            out_path("transfer_matrix_detail.csv"), index=False)

    # Discovered formulas JSON
    formulas_log = {}
    for dname, expr in discovered_exprs.items():
        formulas_log[dname] = {
            "formula": str(expr),
            "best_fraction": best_formulas.get(dname, {}).get(
                "frac_label", "100pct"),
            "abstraction_r": abstraction_corrs.get(dname, None),
        }
    with open(out_path("discovered_formulas.json"), "w") as f:
        json.dump(formulas_log, f, indent=2)

    # Efficiency frontier plot
    plot_efficiency_frontier(all_results)

    # Main benchmark bar chart
    plot_main_benchmark_bars(all_results)

    # --- Raw per-seed results ---
    raw_rows = []
    for dname, results in all_results.items():
        for r in results:
            for si, seed in enumerate(SEEDS):
                if si < len(r["aucs"]):
                    raw_rows.append({
                        "Dataset": dname, "Model": r["label"],
                        "Seed": seed,
                        "AUC": r["aucs"][si],
                        "Acc": r["accs"][si],
                        "F1": r["f1s"][si],
                    })
    pd.DataFrame(raw_rows).to_csv(
        out_path("raw_per_seed_results.csv"), index=False)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  NSAD-BENCH COMPLETE")
    print(f"  Output folder: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    print(f"\n  DATASETS ({len(dataset_cache)}):")
    for dname in dataset_cache:
        X, y = dataset_cache[dname]
        cfg = DATASETS_CONFIG[dname]
        print(f"    {dname:<20} n={len(X):>7} d={X.shape[1]:>3} "
              f"[{cfg['domain']}/{cfg['geometry']}]")

    print(f"\n  DISCOVERED FORMULAS:")
    for dname, expr in discovered_exprs.items():
        frac = best_formulas.get(dname, {}).get("frac_label", "100pct")
        r_val = abstraction_corrs.get(dname, "N/A")
        if isinstance(r_val, float) and not np.isnan(r_val):
            r_str = f"{r_val:.4f}"
        else:
            r_str = str(r_val)
        print(f"    {dname:<20} [{frac}] sigma(x) = {str(expr)[:50]}")
        print(f"    {'':>20} abstraction r = {r_str}")

    print(f"\n  KEY RESULTS (Specialist vs best standard baseline):")
    for dname, results in all_results.items():
        specialist = next(
            (r for r in results if "Specialist" in r["label"]), None)
        baselines_only = [
            r for r in results if r["label"] in STANDARD_ACTIVATIONS]
        if specialist and baselines_only:
            best_bl = max(baselines_only, key=lambda r: r["auc_mean"])
            delta = specialist["auc_mean"] - best_bl["auc_mean"]
            arrow = "UP" if delta > 0 else "DOWN"
            print(f"    {dname:<20} "
                  f"Specialist={specialist['auc_mean']:.4f} "
                  f"vs {best_bl['label']}={best_bl['auc_mean']:.4f} "
                  f"({arrow} {abs(delta):.4f})")

    if stochasticity_results:
        print(f"\n  PySR STOCHASTICITY:")
        for dname, runs in stochasticity_results.items():
            unique_forms = len(set(r["formula"][:30] for r in runs))
            stability = 1 - (unique_forms - 1) / max(len(runs) - 1, 1)
            print(f"    {dname}: {len(runs)} runs -> "
                  f"{unique_forms} unique forms "
                  f"(stability={stability:.0%})")

    print(f"\n  OUTPUT FILES (in {OUTPUT_DIR}/):")
    expected_outputs = [
        "config.json",
        "benchmark_log.txt",
        "dataset_summary.csv",
        "nsad_bench_results.csv",
        "raw_per_seed_results.csv",
        "significance_tests.csv",
        "fraction_sweep_results.csv",
        "fraction_sweep.png", "fraction_sweep.pdf",
        "transfer_matrix.csv", "transfer_matrix_detail.csv",
        "transfer_heatmap.png", "transfer_heatmap.pdf",
        "spectral_full.png", "spectral_full.pdf",
        "spectral_cosine.csv", "spectral_kl.csv",
        "sr_stochasticity.csv",
        "sr_stochasticity.png", "sr_stochasticity.pdf",
        "constrained_sr_formulas.csv",
        "abstraction_correlations.csv",
        "efficiency_frontier.png", "efficiency_frontier.pdf",
        "main_benchmark.png", "main_benchmark.pdf",
        "depth_ablation.png", "depth_ablation.pdf",
        "component_ablation.png", "component_ablation.pdf",
        "discovered_formulas.json",
    ]
    # Add per-dataset manual_vs_sr plots
    for dname in manual_results_all:
        expected_outputs.append(f"manual_vs_sr_{dname}.png")

    for fn in sorted(set(expected_outputs)):
        full_path = OUTPUT_DIR / fn
        exists = "OK" if full_path.exists() else "--"
        print(f"    {exists} {fn}")

    print(f"\n{'=' * 70}")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")

    # Restore stdout
    sys.stdout = logger.terminal
    logger.close()

    # Final console confirmation
    print(f"\n  NSAD-Bench Complete. Results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    t_start = time.time()
    main()
    elapsed_h = (time.time() - t_start) / 3600
    print(f"  Wall time: {elapsed_h:.1f} hours")
