"""
plot_results.py
───────────────
Generate publication-quality charts (LIGHT theme, 300 dpi)
for the research paper:
  "AI Classification and Mapping of CTI into MITRE ATT&CK Techniques"

Models compared (7):
  1. BERT          (results/BErt)
  2. DistilBERT    (results/DistilBERT)
  3. CySecBERT     (results/CySecBERT results)
  4. ModernBERT    (results/MordernBert)
  5. SecBERT       (results/SecBERT result)
  6. SecureBERT    (results/secureBert)
  7. TextCNN       (results/TextCNN)

Datasets : CTI-MITRE  |  Joint  |  TRAM
Scenarios: A0 (baseline), G0 (GenericEDA-1st), B1 (CyberEDA-1st),
           G1 (GenericEDA-2nd), B2 (CyberEDA-2nd / B2_E1)

Figures produced (saved to results/figures/):
  Fig 1  – Macro-F1 grouped bar (1×3 datasets)
  Fig 2  – Macro-Recall grouped bar
  Fig 3  – Micro-F1 grouped bar
  Fig 4  – Weighted-F1 grouped bar
  Fig 5  – Hit@3 & Hit@5 (2×3 grid)
  Fig 6  – MRR & MAP (2×3 grid)
  Fig 7  – Average Training Time per Dataset (grouped bar)
  Fig 8  – Baseline A0 vs Best-EDA Δ improvement
  Fig 9  – 4-Tier Frequency-Group Mean-F1 (grouped bar per dataset, best scenario)
  Fig 10 – Number of Missed Labels (Zero-F1 Techniques) per Dataset & Scenario

Also saves:
  results/figures/model_comparison_summary.csv
  results/figures/frequency_group_summary.csv
  results/figures/missed_labels_summary.csv
"""

import json
import os
import sys
import warnings
from pathlib import Path

# ── UTF-8 on Windows ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8.5,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG & PATHS
# ═════════════════════════════════════════════════════════════════════════════
ROOT         = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"
FIG_DIR      = RESULTS_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["BERT", "DistilBERT", "CySecBERT", "ModernBERT", "SecBERT", "SecureBERT", "TextCNN"]

DATASETS = ["cti-mitre", "joint", "tram"]
DATASET_LABELS = {
    "cti-mitre": "CTI-MITRE (188 labels)",
    "joint":     "Joint (188 labels)",
    "tram":      "TRAM (50 labels)",
}
DATASET_SHORT = {"cti-mitre": "CTI-MITRE", "joint": "Joint", "tram": "TRAM"}

SCENARIOS = ["A0", "G0", "B1", "G1", "B2"]
SCENARIO_LABELS = {
    "A0": "A0\n(Baseline)",
    "G0": "G0\n(GenEDA-S1)",
    "B1": "B1\n(CybEDA-S1)",
    "G1": "G1\n(GenEDA-S2)",
    "B2": "B2\n(CybEDA-S2)",
}

# ── Colour palette (publication-friendly, distinguishable in greyscale) ──
MODEL_COLORS = {
    "BERT":       "#E69F00",  # Amber
    "DistilBERT": "#56B4E9",  # Sky blue
    "CySecBERT":  "#CC79A7",  # Reddish purple
    "ModernBERT": "#009E73",  # Bluish green
    "SecBERT":    "#D55E00",  # Vermilion
    "SecureBERT": "#0072B2",  # Blue
    "TextCNN":    "#999999",  # Grey (baseline)
}
MODEL_HATCHES = {
    "BERT": "", "DistilBERT": "", "CySecBERT": "",
    "ModernBERT": "", "SecBERT": "", "SecureBERT": "", "TextCNN": "//",
}
MODEL_MARKERS = {
    "BERT": "o", "DistilBERT": "s", "CySecBERT": "D",
    "ModernBERT": "^", "SecBERT": "P", "SecureBERT": "X", "TextCNN": "p",
}

FREQ_GROUPS = ["Head", "Medium", "Tail"]
FREQ_COLORS = {"Head": "#009E73", "Medium": "#E69F00", "Tail": "#D55E00"}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING — handles every folder naming convention
# ═════════════════════════════════════════════════════════════════════════════
def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _first_metrics(roots):
    """Return first metrics.json found recursively under given roots."""
    for root in roots:
        if not root.exists():
            continue
        for m in root.rglob("metrics.json"):
            data = _load_json(m)
            if data is not None:
                return data
    return None


def find_metrics(model: str, dataset: str, scenario: str):
    """Locate metrics.json for (model, dataset, scenario)."""
    ds_aliases = {
        "cti-mitre": ["cti-mitre", "cti_to_mitre", "cti-to-mitre",
                       "cti_mitre", "cti", "mitre"],
        "joint":     ["joint"],
        "tram":      ["tram"],
    }[dataset]

    sc_prefixes = [scenario]
    if scenario == "B2":
        sc_prefixes.extend(["B2_E1", "B2-E1"])

    # — BERT  (results/BErt/<SC>_<DS>/seed_*/<metrics.json>)
    if model == "BERT":
        bdir = RESULTS_ROOT / "BErt"
        if not bdir.exists():
            return None
        for sc_p in sc_prefixes:
            for da in ds_aliases:
                # Old style: A0/Joint/metrics.json
                old = bdir / sc_p / da.title() / "metrics.json"
                if old.exists():
                    return _load_json(old)
                # New style: A0_Joint/seed_42_joint/metrics.json
                for folder in bdir.iterdir():
                    if not folder.is_dir():
                        continue
                    fn = folder.name.lower()
                    if fn.startswith(sc_p.lower()) and any(a in fn for a in ds_aliases):
                        r = _first_metrics([folder])
                        if r is not None:
                            return r
        return None

    # — DistilBERT  (results/DistilBERT/<SC>_<ds>/seed_42/metrics.json)
    if model == "DistilBERT":
        ddir = RESULTS_ROOT / "DistilBERT"
        if not ddir.exists():
            return None
        for sc_p in sc_prefixes:
            for da in ds_aliases:
                for folder in ddir.iterdir():
                    if not folder.is_dir():
                        continue
                    fn = folder.name.lower()
                    if fn.startswith(sc_p.lower()) and any(a in fn for a in ds_aliases):
                        r = _first_metrics([folder])
                        if r is not None:
                            return r
        return None

    # — CySecBERT  (results/CySecBERT results/CySecBERT_<SC>_<ds>/metrics.json)
    if model == "CySecBERT":
        croot = RESULTS_ROOT / "CySecBERT results"
        if not croot.exists():
            return None
        for sc_p in sc_prefixes:
            for sub in croot.iterdir():
                if not sub.is_dir():
                    continue
                fn = sub.name.lower()
                sc_ok = (f"_{sc_p.lower()}_" in fn or fn.endswith(f"_{sc_p.lower()}"))
                ds_ok = any(da in fn for da in ds_aliases)
                if sc_ok and ds_ok:
                    r = _first_metrics([sub])
                    if r is not None:
                        return r
        return None

    # — ModernBERT  (results/MordernBert/<SC>_<ds>/<SC>/seed_42/metrics.json)
    if model == "ModernBERT":
        mdir = RESULTS_ROOT / "MordernBert"
        if not mdir.exists():
            return None
        for sc_p in sc_prefixes:
            for da in ds_aliases:
                for folder in mdir.iterdir():
                    if not folder.is_dir():
                        continue
                    fn = folder.name.lower()
                    if fn.startswith(sc_p.lower()) and any(a in fn for a in ds_aliases):
                        r = _first_metrics([folder])
                        if r is not None:
                            return r
        return None

    # — SecBERT  (results/SecBERT result/SecBERT result file/SecBERT_<SC>_<ds>/metrics.json)
    if model == "SecBERT":
        sdir = RESULTS_ROOT / "SecBERT result"
        if not sdir.exists():
            return None
        for sc_p in sc_prefixes:
            for sub in sdir.rglob("metrics.json"):
                fn = sub.parent.name.lower()
                sc_ok = (f"_{sc_p.lower()}_" in fn or fn.endswith(f"_{sc_p.lower()}"))
                ds_ok = any(da in fn for da in ds_aliases)
                if sc_ok and ds_ok:
                    data = _load_json(sub)
                    if data is not None:
                        return data
        return None

    # — SecureBERT  (results/secureBert/<SC>_<ds>/seed_42/metrics.json)
    if model == "SecureBERT":
        sdir = RESULTS_ROOT / "secureBert"
        if not sdir.exists():
            return None
        for sc_p in sc_prefixes:
            for da in ds_aliases:
                for folder in sdir.iterdir():
                    if not folder.is_dir():
                        continue
                    fn = folder.name.lower()
                    if fn.startswith(sc_p.lower()) and any(a in fn for a in ds_aliases):
                        r = _first_metrics([folder])
                        if r is not None:
                            return r
        return None

    # — TextCNN  (results/TextCNN/results/TextCNN/<SC>_<ds>/seed_42/metrics.json)
    if model == "TextCNN":
        tdir = RESULTS_ROOT / "TextCNN" / "results" / "TextCNN"
        if not tdir.exists():
            tdir = RESULTS_ROOT / "TextCNN"
        for sc_p in sc_prefixes:
            for da in ds_aliases:
                for folder in tdir.iterdir():
                    if not folder.is_dir():
                        continue
                    fn = folder.name.lower()
                    if fn.startswith(sc_p.lower()) and any(a in fn for a in ds_aliases):
                        r = _first_metrics([folder])
                        if r is not None:
                            return r
        return None

    return None


# ═════════════════════════════════════════════════════════════════════════════
# 1b. FREQUENCY-GROUP DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def _find_freq_group_csv(model_dir: Path, scenario: str, dataset: str):
    """Find frequency_group_performance.csv for a given model/scenario/dataset."""
    ds_aliases = {
        "cti-mitre": ["cti-mitre", "cti_to_mitre", "cti-to-mitre",
                       "cti_mitre", "mitre", "_cti"],
        "joint":     ["joint"],
        "tram":      ["tram"],
    }[dataset]
    sc_prefixes = [scenario]
    if scenario == "B2":
        sc_prefixes.extend(["B2_E1", "B2-E1"])

    for fgp in model_dir.rglob("frequency_group_performance.csv"):
        # Walk relative to model_dir to avoid workspace path contamination
        path_str = str(fgp.relative_to(model_dir)).lower()
        sc_ok = any(s.lower() in path_str for s in sc_prefixes)
        ds_ok = any(a in path_str for a in ds_aliases)
        if sc_ok and ds_ok:
            return fgp
    return None


def build_freq_group_df() -> pd.DataFrame:
    """Build a DataFrame with frequency-group metrics for all models/scenarios/datasets."""
    model_dirs = {
        "TextCNN":    RESULTS_ROOT / "TextCNN",
        "CySecBERT":  RESULTS_ROOT / "CySecBERT results",
        "DistilBERT": RESULTS_ROOT / "DistilBERT",
        "ModernBERT": RESULTS_ROOT / "MordernBert",
        "SecureBERT":  RESULTS_ROOT / "secureBert",
        "SecBERT":    RESULTS_ROOT / "SecBERT result",
    }
    rows = []
    for model_name, model_dir in model_dirs.items():
        if not model_dir.exists():
            continue
        for ds in DATASETS:
            for sc in SCENARIOS:
                csv_path = _find_freq_group_csv(model_dir, sc, ds)
                if csv_path is None:
                    continue
                try:
                    fgp = pd.read_csv(csv_path)
                    for _, row in fgp.iterrows():
                        rows.append({
                            "model":    model_name,
                            "dataset":  ds,
                            "scenario": sc,
                            "group":    row["Frequency_Group"],
                            "n_labels": int(row["n_labels"]),
                            "mean_precision":  row["mean_precision"],
                            "mean_recall":     row["mean_recall"],
                            "mean_f1":         row["mean_f1"],
                            "median_f1":       row["median_f1"],
                            "f1_zero_pct":     row["f1_zero_percentage"],
                            "total_test_support": int(row["total_test_support"]),
                        })
                except Exception:
                    pass
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 1c. PER-LABEL MISSED LABELS DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def find_per_label_csv(model_name: str, dataset: str, scenario: str):
    """Locate per-label CSV for (model, dataset, scenario)."""
    ds_aliases = {
        "cti-mitre": ["cti-mitre", "cti_to_mitre", "cti-to-mitre", "cti_mitre", "mitre", "_cti"],
        "joint":     ["joint"],
        "tram":      ["tram"],
    }[dataset]
    sc_prefixes = [scenario]
    if scenario == "B2":
        sc_prefixes.extend(["B2_E1", "B2-E1"])

    mapping = {
        "BERT":       RESULTS_ROOT / "BErt",
        "CySecBERT":  RESULTS_ROOT / "CySecBERT results",
        "DistilBERT": RESULTS_ROOT / "DistilBERT",
        "ModernBERT": RESULTS_ROOT / "MordernBert",
        "SecBERT":    RESULTS_ROOT / "SecBERT result",
        "SecureBERT": RESULTS_ROOT / "secureBert",
        "TextCNN":    RESULTS_ROOT / "TextCNN",
    }
    mdir = mapping.get(model_name)
    if not mdir or not mdir.exists():
        return None

    candidates = []
    for p in mdir.rglob("*.csv"):
        # Match relative to mdir to avoid workspace path contamination
        plow = str(p.relative_to(mdir)).lower()
        if any(x in p.name.lower() for x in ["per_label_global", "per_label_metrics", "per_label"]):
            sc_ok = any(s.lower() in plow for s in sc_prefixes)
            ds_ok = any(d in plow for d in ds_aliases)
            if sc_ok and ds_ok:
                candidates.append(p)
    for c in candidates:
        if "per_label_global" in c.name.lower():
            return c
    for c in candidates:
        if "per_label_metrics" in c.name.lower():
            return c
    return candidates[0] if candidates else None


def extract_missed_labels(per_label_path: Path):
    """Extract count and percentage of labels in test set where F1 == 0 (missed labels)."""
    try:
        df_pl = pd.read_csv(per_label_path)
        sup_col = [c for c in df_pl.columns if "test_support" in c.lower() or ("support" in c.lower() and "test" in c.lower())][0]
        f1_col = [c for c in df_pl.columns if c.lower() == "f1"][0]
        test_present = df_pl[df_pl[sup_col] > 0]
        total_present = len(test_present)
        missed = int((test_present[f1_col] == 0).sum())
        pct = float(missed / total_present * 100) if total_present > 0 else 0.0
        return total_present, missed, pct
    except Exception:
        return np.nan, np.nan, np.nan


# ═════════════════════════════════════════════════════════════════════════════
# 2.  BUILD MAIN DATAFRAME
# ═════════════════════════════════════════════════════════════════════════════
def build_df() -> pd.DataFrame:
    rows = []
    for model_name in MODELS:
        for ds in DATASETS:
            for sc in SCENARIOS:
                m = find_metrics(model_name, ds, sc)
                if m is None:
                    print(f"  [WARN] Missing: {model_name:12s} / {ds:10s} / {sc}")
                    continue
                tg = m.get("test_global", {})

                # Also extract missed label counts from per-label CSV
                pl_csv = find_per_label_csv(model_name, ds, sc)
                tot_lbls, missed_lbls, missed_pct = (np.nan, np.nan, np.nan)
                if pl_csv:
                    tot_lbls, missed_lbls, missed_pct = extract_missed_labels(pl_csv)

                rows.append({
                    "model":            model_name,
                    "dataset":          ds,
                    "scenario":         sc,
                    "macro_f1":         tg.get("macro_f1",         np.nan),
                    "macro_recall":     tg.get("macro_recall",     np.nan),
                    "macro_precision":  tg.get("macro_precision",  np.nan),
                    "micro_f1":         tg.get("micro_f1",         np.nan),
                    "micro_recall":     tg.get("micro_recall",     np.nan),
                    "micro_precision":  tg.get("micro_precision",  np.nan),
                    "weighted_f1":      tg.get("weighted_f1",      np.nan),
                    "hit_at_3":         tg.get("hit_at_3",         np.nan),
                    "hit_at_5":         tg.get("hit_at_5",         np.nan),
                    "mrr":              tg.get("mrr",              np.nan),
                    "map":              tg.get("map",              np.nan),
                    "hamming_loss":     tg.get("hamming_loss",     np.nan),
                    "train_sec":        m.get("training_seconds",  np.nan),
                    "vram_mb":          m.get("peak_vram_mb",      np.nan),
                    "total_test_labels": tot_lbls,
                    "missed_labels":    missed_lbls,
                    "missed_pct":       missed_pct,
                })
    return pd.DataFrame(rows)


print("=" * 80)
print("  Loading metrics for all 7 models ...")
print("=" * 80)
df = build_df()
print(f"\n  Loaded {len(df)} records  |  "
      f"{df['model'].nunique()} models  |  "
      f"{df['dataset'].nunique()} datasets  |  "
      f"{df['scenario'].nunique()} scenarios\n")

csv_path = FIG_DIR / "model_comparison_summary.csv"
df.to_csv(csv_path, index=False)
print(f"  Summary CSV saved -> {csv_path}")

ml_csv = FIG_DIR / "missed_labels_summary.csv"
df[["model", "dataset", "scenario", "total_test_labels", "missed_labels", "missed_pct"]].to_csv(ml_csv, index=False)
print(f"  Missed labels CSV saved -> {ml_csv}")

print("\n  Loading frequency-group data ...")
df_fg = build_freq_group_df()
if not df_fg.empty:
    fg_csv = FIG_DIR / "frequency_group_summary.csv"
    df_fg.to_csv(fg_csv, index=False)
    print(f"  Freq-group CSV saved -> {fg_csv}  ({len(df_fg)} rows)")
else:
    print("  [WARN] No frequency-group data found.")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  SHARED HELPERS  (LIGHT THEME)
# ═════════════════════════════════════════════════════════════════════════════
num_models = len(MODELS)
BAR_W      = 0.11
X_IDX      = np.arange(len(SCENARIOS))


def save(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved -> {path.name}")
    plt.close(fig)


def style_ax(ax, grid_axis="y"):
    """Apply consistent light theme."""
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.6, linestyle="--", zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")


def add_bar_labels(ax, bars, fmt=".3f", pad=0.005, fontsize=5.5,
                   suffix="", rotation=90, color="#333333"):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            f"{h:{fmt}}{suffix}",
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold",
            rotation=rotation,
        )


def dark_legend(ax, **kw):
    leg = ax.legend(framealpha=0.85, facecolor="white",
                    edgecolor="#CCCCCC", **kw)
    return leg


def grouped_bar_figure(metric_col: str, ylabel: str, title: str,
                       fname: str, y_floor=0.20, y_ceil=1.0):
    """Generic 1×3 grouped-bar figure for a single metric."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5), sharey=False)
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes, DATASETS):
        style_ax(ax)
        sub = df[df["dataset"] == ds]

        for i, model in enumerate(MODELS):
            vals = []
            for sc in SCENARIOS:
                v = sub[(sub["model"] == model) & (sub["scenario"] == sc)][metric_col].values
                vals.append(v[0] if len(v) else np.nan)
            offset = (i - (num_models - 1) / 2) * BAR_W
            bars = ax.bar(X_IDX + offset, vals, BAR_W, label=model,
                          color=MODEL_COLORS[model], alpha=0.88,
                          edgecolor="white", linewidth=0.4,
                          hatch=MODEL_HATCHES.get(model, ""), zorder=3)
            add_bar_labels(ax, bars, pad=0.006, fontsize=5)

        ax.set_title(DATASET_SHORT[ds], fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(X_IDX)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=7.5)
        ds_vals = sub[metric_col].dropna()
        lo = max(y_floor, ds_vals.min() - 0.07) if len(ds_vals) else y_floor
        hi = min(y_ceil,  ds_vals.max() + 0.10) if len(ds_vals) else y_ceil
        ax.set_ylim(lo, hi)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        dark_legend(ax, loc="upper left", fontsize=7)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, fname)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1–4: Main Metric Grouped Bars
# ═════════════════════════════════════════════════════════════════════════════
print("\nFig 1 - Macro-F1 Comparison")
grouped_bar_figure("macro_f1", "Macro-F1", "Macro-F1 Across 7 Models × 5 Scenarios",
                   "fig01_macro_f1.png", y_floor=0.20)

print("Fig 2 - Macro-Recall Comparison")
grouped_bar_figure("macro_recall", "Macro-Recall", "Macro-Recall Across 7 Models × 5 Scenarios",
                   "fig02_macro_recall.png", y_floor=0.15)

print("Fig 3 - Micro-F1 Comparison")
grouped_bar_figure("micro_f1", "Micro-F1", "Micro-F1 Across 7 Models × 5 Scenarios",
                   "fig03_micro_f1.png", y_floor=0.35)

print("Fig 4 - Weighted-F1 Comparison")
grouped_bar_figure("weighted_f1", "Weighted-F1", "Weighted-F1 Across 7 Models × 5 Scenarios",
                   "fig04_weighted_f1.png", y_floor=0.35)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Hit@3 & Hit@5 (2×3 grid)
# ═════════════════════════════════════════════════════════════════════════════
print("Fig 5 - Hit@3 & Hit@5")

fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharey=False)
fig.patch.set_facecolor("white")

for row_idx, (metric_col, row_label) in enumerate(
        [("hit_at_3", "Hit@3"), ("hit_at_5", "Hit@5")]):
    for col_idx, ds in enumerate(DATASETS):
        ax = axes[row_idx][col_idx]
        style_ax(ax)
        sub = df[df["dataset"] == ds]

        for i, model in enumerate(MODELS):
            vals = []
            for sc in SCENARIOS:
                v = sub[(sub["model"] == model) & (sub["scenario"] == sc)][metric_col].values
                vals.append(v[0] if len(v) else np.nan)
            offset = (i - (num_models - 1) / 2) * BAR_W
            bars = ax.bar(X_IDX + offset, vals, BAR_W, label=model,
                          color=MODEL_COLORS[model], alpha=0.88,
                          edgecolor="white", linewidth=0.4, zorder=3)
            add_bar_labels(ax, bars, pad=0.005, fontsize=5)

        ax.set_title(f"{row_label} — {DATASET_SHORT[ds]}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(X_IDX)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=7)
        ds_vals = sub[metric_col].dropna()
        lo = max(0.50, ds_vals.min() - 0.05) if len(ds_vals) else 0.50
        hi = min(1.00, ds_vals.max() + 0.07) if len(ds_vals) else 1.00
        ax.set_ylim(lo, hi)
        ax.set_ylabel(row_label, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        dark_legend(ax, loc="lower right", fontsize=6.5)

fig.suptitle("Ranking Metrics: Hit@3 & Hit@5", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "fig05_hit3_hit5.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6: MRR & MAP (2×3 grid)
# ═════════════════════════════════════════════════════════════════════════════
print("Fig 6 - MRR & MAP")

fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharey=False)
fig.patch.set_facecolor("white")

for row_idx, (metric_col, row_label) in enumerate([("mrr", "MRR"), ("map", "MAP")]):
    for col_idx, ds in enumerate(DATASETS):
        ax = axes[row_idx][col_idx]
        style_ax(ax)
        sub = df[df["dataset"] == ds]
        for i, model in enumerate(MODELS):
            vals = []
            for sc in SCENARIOS:
                v = sub[(sub["model"] == model) & (sub["scenario"] == sc)][metric_col].values
                vals.append(v[0] if len(v) else np.nan)
            offset = (i - (num_models - 1) / 2) * BAR_W
            bars = ax.bar(X_IDX + offset, vals, BAR_W, label=model,
                          color=MODEL_COLORS[model], alpha=0.88,
                          edgecolor="white", linewidth=0.4, zorder=3)
            add_bar_labels(ax, bars, pad=0.005, fontsize=5)
        ax.set_title(f"{row_label} — {DATASET_SHORT[ds]}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(X_IDX)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=7)
        ds_vals = sub[metric_col].dropna()
        lo = max(0.50, ds_vals.min() - 0.05) if len(ds_vals) else 0.50
        hi = min(1.00, ds_vals.max() + 0.07) if len(ds_vals) else 1.00
        ax.set_ylim(lo, hi)
        ax.set_ylabel(row_label, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        dark_legend(ax, loc="lower right", fontsize=6.5)

fig.suptitle("Ranking Quality: MRR & MAP", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "fig06_mrr_map.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Average Training Time per Dataset
# ═════════════════════════════════════════════════════════════════════════════
print("Fig 7 - Training Time Comparison")

fig, ax_bar = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("white")
style_ax(ax_bar)

ds_x  = np.arange(len(DATASETS))
ds_bw = 0.11

for i, model in enumerate(MODELS):
    avg_times = [
        df[(df["model"] == model) & (df["dataset"] == ds)]["train_sec"].mean()
        for ds in DATASETS
    ]
    offset = (i - (num_models - 1) / 2) * ds_bw
    bars = ax_bar.bar(ds_x + offset, avg_times, ds_bw, label=model,
                      color=MODEL_COLORS[model], alpha=0.88,
                      edgecolor="white", linewidth=0.4, zorder=3)
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 10,
                        f"{h:.0f}s", ha="center", va="bottom",
                        fontsize=6.5, color="#333", fontweight="bold")

ax_bar.set_title("Average Training Time per Dataset (Lower = Faster)",
                 fontsize=13, fontweight="bold", pad=12)
ax_bar.set_xticks(ds_x)
ax_bar.set_xticklabels([DATASET_SHORT[d] for d in DATASETS], fontsize=11)
ax_bar.set_ylabel("Training Time (s)", fontsize=11)
dark_legend(ax_bar, loc="upper right", fontsize=8.5)

plt.tight_layout()
save(fig, "fig07_training_time.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Baseline A0 vs Best EDA — Δ Improvement
# ═════════════════════════════════════════════════════════════════════════════
print("Fig 8 - Baseline A0 vs Best Augmentation (Macro-F1)")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor("white")

x_m = np.arange(num_models)
for ax, ds in zip(axes, DATASETS):
    style_ax(ax)
    sub = df[df["dataset"] == ds]

    baseline_vals, best_aug_vals, best_aug_sc = [], [], []
    for model in MODELS:
        m_sub = sub[sub["model"] == model]
        bv = m_sub[m_sub["scenario"] == "A0"]["macro_f1"].values
        baseline_vals.append(bv[0] if len(bv) else np.nan)
        aug_sub = m_sub[m_sub["scenario"].isin(["B1", "B2", "G0", "G1"])]
        if aug_sub.empty or aug_sub["macro_f1"].isna().all():
            best_aug_vals.append(np.nan)
            best_aug_sc.append("")
        else:
            best_row = aug_sub.loc[aug_sub["macro_f1"].idxmax()]
            best_aug_vals.append(best_row["macro_f1"])
            best_aug_sc.append(best_row["scenario"])

    bw = 0.32
    ax.bar(x_m - bw / 2, baseline_vals, bw, label="A0 (Baseline)",
           color="#BBBBBB", edgecolor="white", linewidth=0.4, zorder=3)
    bars_best = ax.bar(x_m + bw / 2, best_aug_vals, bw, label="Best EDA",
                       color=[MODEL_COLORS[m] for m in MODELS],
                       edgecolor="white", linewidth=0.4, zorder=3)

    # Δ labels
    for xi, (b, a, lbl) in enumerate(zip(baseline_vals, best_aug_vals, best_aug_sc)):
        if not np.isnan(b) and not np.isnan(a):
            delta = a - b
            sign = "+" if delta >= 0 else ""
            ax.annotate(f"{lbl}\n{sign}{delta:.3f}",
                        xy=(xi + bw / 2, a), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6.5,
                        fontweight="bold", color="#333")

    ax.set_title(DATASET_SHORT[ds], fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x_m)
    ax.set_xticklabels(MODELS, fontsize=8, rotation=15)
    ds_vals = sub["macro_f1"].dropna()
    lo = max(0.15, ds_vals.min() - 0.06) if len(ds_vals) else 0.15
    hi = min(1.00, ds_vals.max() + 0.12) if len(ds_vals) else 1.00
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Macro-F1", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    dark_legend(ax, loc="upper left")

fig.suptitle("Baseline (A0) vs Best EDA Augmentation — Macro-F1 Improvement",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "fig08_baseline_vs_best.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9: 4-Tier Frequency Group Mean-F1 (Best Scenario, per dataset)
# ═════════════════════════════════════════════════════════════════════════════
if not df_fg.empty:
    print("Fig 9 - 4-Tier Frequency-Group Mean-F1 (Best Scenario)")

    # For each model+dataset, pick the scenario with best overall mean_f1
    fg_models = [m for m in MODELS if m in df_fg["model"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes, DATASETS):
        style_ax(ax)
        sub_fg = df_fg[df_fg["dataset"] == ds]
        if sub_fg.empty:
            ax.set_title(f"{DATASET_SHORT[ds]} (no data)")
            continue

        x_grp = np.arange(len(FREQ_GROUPS))
        n_fg_models = len(fg_models)
        bw = 0.12

        for i, model in enumerate(fg_models):
            m_sub = sub_fg[sub_fg["model"] == model]
            if m_sub.empty:
                continue
            # Find best scenario (highest average mean_f1 across groups)
            best_sc = m_sub.groupby("scenario")["mean_f1"].mean().idxmax()
            best = m_sub[m_sub["scenario"] == best_sc]

            vals = []
            for grp in FREQ_GROUPS:
                g_row = best[best["group"] == grp]
                vals.append(g_row["mean_f1"].values[0] if len(g_row) else np.nan)

            offset = (i - (n_fg_models - 1) / 2) * bw
            bars = ax.bar(x_grp + offset, vals, bw,
                          label=f"{model} ({best_sc})",
                          color=MODEL_COLORS.get(model, "#999"),
                          edgecolor="white", linewidth=0.4, zorder=3)
            add_bar_labels(ax, bars, pad=0.008, fontsize=6)

        ax.set_title(DATASET_SHORT[ds], fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(x_grp)
        ax.set_xticklabels(FREQ_GROUPS, fontsize=11)
        ax.set_ylabel("Mean F1 Score", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        dark_legend(ax, loc="upper right", fontsize=7)

    fig.suptitle("Frequency-Group Mean-F1 by Model (Best Scenario)\n"
                 "Head (≥100 samples)  |  Medium (20–99)  |  Tail (<20)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    save(fig, "fig09_freq_group_mean_f1.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Count of Missed Labels (Zero-F1 Techniques) by Model and Scenario
# ═════════════════════════════════════════════════════════════════════════════
print("Fig 10 - Count of Missed Labels (Zero-F1 Techniques)")
fig, axes = plt.subplots(1, 3, figsize=(24, 6.5), sharey=False)
fig.patch.set_facecolor("white")

for ax, ds in zip(axes, DATASETS):
    style_ax(ax)
    sub = df[df["dataset"] == ds]

    for i, model in enumerate(MODELS):
        vals = []
        for sc in SCENARIOS:
            v = sub[(sub["model"] == model) & (sub["scenario"] == sc)]["missed_labels"].values
            vals.append(v[0] if len(v) else np.nan)
        offset = (i - (num_models - 1) / 2) * BAR_W
        bars = ax.bar(X_IDX + offset, vals, BAR_W, label=model,
                      color=MODEL_COLORS[model], alpha=0.88,
                      edgecolor="white", linewidth=0.4,
                      hatch=MODEL_HATCHES.get(model, ""), zorder=3)
        add_bar_labels(ax, bars, fmt=".0f", pad=0.8, fontsize=5.5)

    ax.set_title(DATASET_SHORT[ds], fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(X_IDX)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=7.5)
    ax.set_ylabel("Missed Labels Count (Zero-F1)", fontsize=10)

    max_val = sub["missed_labels"].dropna().max() if not sub["missed_labels"].dropna().empty else 10
    ax.set_ylim(0, max_val * 1.18 + 2)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    dark_legend(ax, loc="upper right", fontsize=7.5)

fig.suptitle("Figure 10: Number of Missed MITRE ATT&CK Techniques (F1 = 0) by Model and Scenario\n"
             "Lower is better — Evaluating Technique Suppression & Long-Tail Recovery across Baseline vs. Augmentations",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
save(fig, "fig10_missed_labels.png")



# ═════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  SUMMARY: Average Metrics per Model (all datasets & scenarios)")
print("=" * 80)

summary = df.groupby("model").agg(
    avg_macro_f1    = ("macro_f1",     "mean"),
    best_macro_f1   = ("macro_f1",     "max"),
    avg_micro_f1    = ("micro_f1",     "mean"),
    avg_weighted_f1 = ("weighted_f1",  "mean"),
    avg_mrr         = ("mrr",          "mean"),
    avg_train_sec   = ("train_sec",    "mean"),
    avg_vram_mb     = ("vram_mb",      "mean"),
).reset_index().sort_values("avg_macro_f1", ascending=False)

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)
print(summary.to_string(index=False))

# ── Per-dataset best result table ──
print("\n  Best Macro-F1 per Model per Dataset:")
print("  " + "-" * 70)
for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    best_per_model = sub.loc[sub.groupby("model")["macro_f1"].idxmax()][
        ["model", "scenario", "macro_f1"]
    ].sort_values("macro_f1", ascending=False)
    print(f"\n  [{DATASET_SHORT[ds]}]")
    for _, row in best_per_model.iterrows():
        print(f"    {row['model']:12s}  {row['scenario']:5s}  Macro-F1 = {row['macro_f1']:.4f}")

print("\n" + "=" * 80)
print(f"  All figures saved to: {FIG_DIR.resolve()}")
print("  Figures generated:")
for f in sorted(FIG_DIR.glob("fig*.png")):
    print(f"    {f.name}")
print("=" * 80 + "\n")
