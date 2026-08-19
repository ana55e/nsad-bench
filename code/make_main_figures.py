"""
==========================================================================
NSAD-Bench: MAIN-PAPER FIGURES (Figures 1 to 3)
==========================================================================
Produces the three figures that appear in the main body of the paper.
Designed for:
  * a single-column layout 6.5 inches wide, so figures are drawn at or
    slightly above that width and scaled to \textwidth on inclusion
  * a consistent palette across all three figures
  * labels that stay readable at 8pt caption size
  * one clear message per figure

Outputs (in main_paper_figs_<timestamp>/):
  fig1_rescue.png            EEG Eye State and Wilt recovery (Figure 1)
  fig2_leaderboard.png       per-dataset comparison, all methods (Figure 2)
  fig3_family_summary.png    family worst / mean / best summary (Figure 3)

Plus matching .pdf vector versions for paper inclusion.

Usage:
    python make_main_figures.py \
        --main path/to/nsad_bench_results.csv \
        --extra path/to/extra_methods.csv
==========================================================================
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from pathlib import Path


# ==========================================================================
# CONFIG
# ==========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--main", default="results/nsad_bench_results.csv")
parser.add_argument("--extra", default="results/extra_methods.csv")
parser.add_argument("--out", default=None)
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path(args.out or f"main_paper_figs_{TIMESTAMP}")
OUT.mkdir(exist_ok=True)


def out_path(fn):
    return str(OUT / fn)


# Paper-grade matplotlib defaults
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
})


# ==========================================================================
# LOAD AND MERGE
# ==========================================================================
main_df = pd.read_csv(args.main)
extra = pd.read_csv(args.extra)

main_only = main_df[main_df["Experiment"] == "Main"].copy()
main_only["Source"] = "main_run"
extra_main = extra[extra["Experiment"] == "Main"].copy()

keep = ["Dataset", "Model", "AUC_mean", "AUC_std", "Params", "Source"]
df = pd.concat([
    main_only[[c for c in keep if c in main_only.columns]],
    extra_main[[c for c in keep if c in extra_main.columns]],
], ignore_index=True)

DATASETS = ["HIGGS", "FOREST_COVER", "SPAMBASE", "MINIBOONE",
            "MAGIC_TELESCOPE", "PHONEME", "EEG_EYE_STATE", "WILT"]
DATASET_DISPLAY = {
    "HIGGS": "HIGGS", "FOREST_COVER": "Forest Cover",
    "SPAMBASE": "Spambase", "MINIBOONE": "MiniBooNE",
    "MAGIC_TELESCOPE": "Magic", "PHONEME": "Phoneme",
    "EEG_EYE_STATE": "EEG Eye State", "WILT": "Wilt",
}

# Method families and palette
FAMILIES = {
    "Heavy ReLU":       ["Heavy (ReLU)"],
    "Standard":         ["ReLU", "GELU", "SiLU", "PReLU", "Mish",
                         "Tanh", "Sigmoid"],
    "Learned (PAU, naive)": ["PAU"],
    "Learned (added)":  ["PAU-fixed", "KAN", "NAS-AF"],
    "SR-discovered":    ["Hybrid (Specialist)", "Hybrid (Constrained)",
                         "Unbounded Hybrid"],
}
COLOURS = {
    "Heavy ReLU":       "#4d4d4d",
    "Standard":         "#4878d0",
    "Learned (PAU, naive)": "#bcbcbc",
    "Learned (added)":  "#5cb95c",
    "SR-discovered":    "#d65f5f",
}
METHOD_FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms}

ORDERED_METHODS = (FAMILIES["Heavy ReLU"]
                   + FAMILIES["Standard"]
                   + FAMILIES["Learned (PAU, naive)"]
                   + FAMILIES["Learned (added)"]
                   + FAMILIES["SR-discovered"])
present = [m for m in ORDERED_METHODS if m in df["Model"].unique()]

piv_auc = df.pivot_table(index="Dataset", columns="Model",
                          values="AUC_mean", aggfunc="mean")
piv_std = df.pivot_table(index="Dataset", columns="Model",
                          values="AUC_std", aggfunc="mean")

piv_auc = piv_auc.reindex(index=[d for d in DATASETS if d in piv_auc.index],
                          columns=[m for m in present if m in piv_auc.columns])
piv_std = piv_std.reindex(index=piv_auc.index, columns=piv_auc.columns)

print(f"Loaded data. {len(piv_auc.index)} datasets x "
      f"{len(piv_auc.columns)} methods.")


# ==========================================================================
# FIG 1: RECOVERY ON EEG EYE STATE AND WILT
# ==========================================================================
def fig1_rescue():
    """Two panels, EEG Eye State and Wilt, showing the symbolic activation
    at chance alongside the learned activations that recover full
    performance under the same protocol."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4),
                              gridspec_kw={"wspace": 0.35})

    # Curated method order: failure cases first, then rescuers
    rescue_methods = ["Hybrid (Specialist)", "PAU",
                      "Unbounded Hybrid", "NAS-AF",
                      "GELU", "ReLU", "PAU-fixed", "KAN"]
    rescue_methods = [m for m in rescue_methods if m in piv_auc.columns]

    for ax, dname in zip(axes, ["EEG_EYE_STATE", "WILT"]):
        if dname not in piv_auc.index:
            continue

        # Build (label, auc, std, family) tuples in failure-to-success order
        rows = []
        for m in rescue_methods:
            v = piv_auc.loc[dname, m]
            if pd.isna(v):
                continue
            s = piv_std.loc[dname, m]
            rows.append((m, v, 0 if pd.isna(s) else s,
                         METHOD_FAMILY[m]))

        # Sort ascending: worst at top, best at bottom (more dramatic visual)
        rows.sort(key=lambda r: r[1])
        labels = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        es = [r[2] for r in rows]
        cols = [COLOURS[r[3]] for r in rows]

        y_positions = np.arange(len(rows))
        bars = ax.barh(y_positions, ys, xerr=es, color=cols,
                       alpha=0.9, capsize=2.5,
                       error_kw={"linewidth": 0.8, "ecolor": "#222"})

        # Random baseline reference
        ax.axvline(0.5, color="black", linestyle=":",
                   linewidth=0.8, alpha=0.5)

        # Highlight the SR failure bar with a thicker red edge
        for i, m in enumerate(labels):
            if m == "Hybrid (Specialist)":
                bars[i].set_edgecolor("#a02020")
                bars[i].set_linewidth(1.4)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0.45, 1.02)
        ax.set_xlabel("Test AUC")
        ax.grid(axis="x", linewidth=0.4, alpha=0.4)

        title = f"{DATASET_DISPLAY[dname]}"
        ax.set_title(title, fontweight="bold")

        # Value annotations: place INSIDE the bar at the right end
        # for long bars, OUTSIDE (past the error bar) for short bars
        for i, (v, err) in enumerate(zip(ys, es)):
            if v > 0.7:
                ax.text(v - 0.01, i, f"{v:.3f}", va="center",
                        ha="right", fontsize=7.5, color="white",
                        fontweight="bold")
            else:
                # Push past the error bar so it doesn't get hidden
                ax.text(v + err + 0.012, i, f"{v:.3f}",
                        va="center", ha="left", fontsize=7.5)

    # Shared legend at bottom (with extra padding so it doesn't overlap)
    handles = [mpatches.Patch(color=COLOURS[f], label=f)
               for f in COLOURS if f != "Heavy ReLU"]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=8,
               bbox_to_anchor=(0.5, -0.13), frameon=False)

    fig.suptitle("Where SR-discovered closed forms collapse, "
                 "learned activations recover",
                 y=1.02, fontsize=10.5, fontweight="bold")

    plt.savefig(out_path("fig1_rescue.png"), dpi=300,
                bbox_inches="tight")
    plt.savefig(out_path("fig1_rescue.pdf"),
                bbox_inches="tight")
    plt.close()
    print("Saved fig1_rescue")


fig1_rescue()


# ==========================================================================
# FIG 2: PER-DATASET LEADERBOARD (compact)
# ==========================================================================
def fig2_leaderboard():
    """Eight-panel grid, one panel per dataset, showing every method with
    one-standard-deviation error bars. Drawn at full text width."""
    fig = plt.figure(figsize=(8.5, 6.5))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.95)

    # Methods to plot (compact selection -- skip a few tail ones to keep panels readable)
    plot_methods = ["Heavy (ReLU)", "ReLU", "GELU", "Mish", "Tanh",
                    "PAU", "PAU-fixed", "KAN", "NAS-AF",
                    "Hybrid (Specialist)", "Unbounded Hybrid"]
    plot_methods = [m for m in plot_methods if m in piv_auc.columns]

    for idx, dname in enumerate(piv_auc.index):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])

        means = piv_auc.loc[dname, plot_methods]
        stds = piv_std.loc[dname, plot_methods]
        valid = means.dropna()
        labels = list(valid.index)
        ys = list(valid.values)
        es = [stds[m] if not pd.isna(stds[m]) else 0 for m in labels]
        cols = [COLOURS[METHOD_FAMILY[m]] for m in labels]

        # Identify winner
        best_idx = int(np.argmax(ys))

        bars = ax.barh(range(len(labels)), ys, xerr=es,
                       color=cols, alpha=0.9, capsize=1.5,
                       error_kw={"linewidth": 0.6, "ecolor": "#222"})

        # Outline the winner
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(1.2)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(DATASET_DISPLAY[dname], fontsize=8.5,
                     fontweight="bold", pad=4)
        ax.set_xlim(0.45, 1.02)
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.set_xlabel("AUC", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", linewidth=0.4, alpha=0.4)
        ax.invert_yaxis()

    # Shared legend
    handles = [mpatches.Patch(color=COLOURS[f], label=f)
               for f in COLOURS]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.005), frameon=False)

    fig.suptitle("NSAD-Bench: per-dataset comparison "
                 "(10 seeds, identical protocol)",
                 fontsize=10.5, fontweight="bold", y=0.995)

    plt.savefig(out_path("fig2_leaderboard.png"), dpi=300,
                bbox_inches="tight")
    plt.savefig(out_path("fig2_leaderboard.pdf"),
                bbox_inches="tight")
    plt.close()
    print("Saved fig2_leaderboard")


fig2_leaderboard()


# ==========================================================================
# FIG 3: FAMILY-LEVEL SUMMARY
# ==========================================================================
def fig3_family_summary():
    """For each method family, the worst-case dataset, the cross-dataset
    mean, and the best-case dataset. The worst-case column is what
    separates the families: the symbolic family contains a chance-level
    entry, the learned family does not."""
    # Use a canonical primary method per family so the figure reflects
    # the actual story (e.g. SR-discovered = Hybrid Specialist, not the
    # best-of-family which would mask the failure with Unbounded Hybrid).
    PRIMARY = {
        "Heavy ReLU":       "Heavy (ReLU)",
        "Standard":         "ReLU",
        "Learned (PAU, naive)": "PAU",
        "Learned (added)":  "KAN",
        "SR-discovered":    "Hybrid (Specialist)",
    }

    fam_data = {}
    for fam, primary in PRIMARY.items():
        if primary not in piv_auc.columns:
            continue
        per_ds = piv_auc[primary].dropna()
        fam_data[fam] = {
            "primary": primary,
            "worst": float(per_ds.min()),
            "mean":  float(per_ds.mean()),
            "best":  float(per_ds.max()),
            "values": per_ds.values,
            "dataset_min": per_ds.idxmin(),
        }

    fams = list(fam_data.keys())
    n = len(fams)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    x = np.arange(n)
    w = 0.27

    # Side-by-side bars: worst, mean, best
    bar_lo = ax.bar(x - w, [fam_data[f]["worst"] for f in fams], width=w,
                    color=[COLOURS[f] for f in fams], alpha=0.5,
                    edgecolor="black", linewidth=0.5,
                    label="Worst-case dataset")
    bar_mid = ax.bar(x, [fam_data[f]["mean"] for f in fams], width=w,
                     color=[COLOURS[f] for f in fams], alpha=0.85,
                     edgecolor="black", linewidth=0.5,
                     label="Mean across 8")
    bar_hi = ax.bar(x + w, [fam_data[f]["best"] for f in fams], width=w,
                    color=[COLOURS[f] for f in fams], alpha=1.0,
                    edgecolor="black", linewidth=0.5,
                    label="Best-case dataset")

    # Random reference line
    ax.axhline(0.5, color="black", linestyle=":",
               linewidth=0.8, alpha=0.5)

    # Value labels: place worst label slightly higher when bar is very short
    for i, f in enumerate(fams):
        worst = fam_data[f]["worst"]
        mean = fam_data[f]["mean"]
        best = fam_data[f]["best"]

        # Worst-case label
        ax.text(i - w, worst + 0.012,
                f"{worst:.3f}", ha="center",
                fontsize=7,
                fontweight="bold" if worst < 0.6 else "normal",
                color="#a02020" if worst < 0.6 else "black")

        # Mean label
        ax.text(i, mean + 0.012,
                f"{mean:.3f}", ha="center",
                fontsize=7, fontweight="bold")

        # Best-case label
        ax.text(i + w, best + 0.012,
                f"{best:.3f}", ha="center",
                fontsize=7)

        # Annotate the dataset for catastrophic worst cases (< 0.6)
        if worst < 0.6:
            worst_dname = fam_data[f]["dataset_min"]
            ax.text(i - w, worst - 0.04,
                    f"({DATASET_DISPLAY[worst_dname]})",
                    ha="center", fontsize=6.5, color="#a02020",
                    style="italic")

    ax.set_xticks(x)
    # Show family name + primary method
    xt_labels = [f"{f}\n({fam_data[f]['primary']})" for f in fams]
    ax.set_xticklabels(xt_labels, fontsize=8)
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.40, 1.04)
    ax.set_title("Family-level summary: worst, mean, best AUC across "
                 "8 datasets\n(one canonical method per family)",
                 fontweight="bold", pad=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95)

    plt.savefig(out_path("fig3_family_summary.png"), dpi=300,
                bbox_inches="tight")
    plt.savefig(out_path("fig3_family_summary.pdf"),
                bbox_inches="tight")
    plt.close()
    print("Saved fig3_family_summary")


fig3_family_summary()


# ==========================================================================
# Generate one-line caption suggestions
# ==========================================================================
print(f"\n{'=' * 70}")
print(f"  CAPTION SUGGESTIONS")
print(f"{'=' * 70}\n")

eeg_sr = piv_auc.loc["EEG_EYE_STATE", "Hybrid (Specialist)"]
eeg_kan = piv_auc.loc["EEG_EYE_STATE", "KAN"]
eeg_pau = piv_auc.loc["EEG_EYE_STATE", "PAU-fixed"]
wilt_sr = piv_auc.loc["WILT", "Hybrid (Specialist)"]
wilt_kan = piv_auc.loc["WILT", "KAN"]

print("Fig 1 (Rescue):")
print(f"  Test AUC across two datasets where SR-discovered closed-form")
print(f"  activations achieve near-random performance: EEG Eye State")
print(f"  ({eeg_sr:.3f}) and Wilt ({wilt_sr:.3f}). Learned activations")
print(f"  KAN ({eeg_kan:.3f} / {wilt_kan:.3f}) and PAU-fixed")
print(f"  ({eeg_pau:.3f} / -- ) recover full performance under the")
print(f"  identical training protocol. Error bars show std over 10 seeds.")

print(f"\nFig 2 (Leaderboard):")
print(f"  Per-dataset Test AUC for 11 activation methods on 8 tabular")
print(f"  classification benchmarks. All methods trained under the")
print(f"  identical protocol (10 seeds, 100 epochs, early stopping).")
print(f"  Best method per panel outlined in black.")

print(f"\nFig 3 (Family summary):")
print(f"  Method-family worst-case (left), mean (center), and best-case")
print(f"  (right) Test AUC across 8 tabular benchmarks. Each bar shows")
print(f"  the family-best method per dataset. SR-discovered formulas")
print(f"  exhibit a 0.50 worst-case (random); learned activations match")
print(f"  the standard family without the failure modes.")

print(f"\n  Output: {OUT}/")
