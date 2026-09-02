"""
plot_results.py
---------------
Vẽ tự động các biểu đồ so sánh kết quả thực nghiệm
DistilBERT vs SecureBERT trên 3 dataset × 5 scenarios.

Output: results/figures/ (PNG, 300 dpi)
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "DejaVu Sans"
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. PATHS & CONFIG
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # project root
RESULTS_ROOT = ROOT / "results"
FIG_DIR = RESULTS_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "DistilBERT": RESULTS_ROOT / "DistilBERT",
    "SecureBERT": RESULTS_ROOT / "secureBert",
}

DATASETS = ["cti-mitre", "joint", "tram"]
DATASET_LABELS = {
    "cti-mitre": "CTI-MITRE",
    "joint":     "Joint",
    "tram":      "TRAM",
}

SCENARIOS = ["A0", "B1", "B2", "G0", "G1"]
SCENARIO_LABELS = {
    "A0": "A0\n(Baseline)",
    "B1": "B1\n(Cyber EDA\n1-stage)",
    "B2": "B2\n(Cyber EDA\n2-stage)",
    "G0": "G0\n(Generic EDA\n1-stage)",
    "G1": "G1\n(Generic EDA\n2-stage)",
}

# Color palette
COLORS = {
    "DistilBERT": {
        "cti-mitre": "#4C72B0",
        "joint":     "#55A868",
        "tram":      "#C44E52",
    },
    "SecureBERT": {
        "cti-mitre": "#8172B2",
        "joint":     "#CCB974",
        "tram":      "#64B5CD",
    },
}

MODEL_COLORS  = {"DistilBERT": "#3A7FBF", "SecureBERT": "#E07B39"}
DATASET_COLORS = {"cti-mitre": "#4C72B0", "joint": "#55A868", "tram": "#C44E52"}

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def find_metrics(model_dir: Path, dataset: str, scenario_code: str) -> dict | None:
    """Tìm file metrics.json tương ứng với (dataset, scenario)."""
    for folder in model_dir.iterdir():
        if not folder.is_dir():
            continue
        folder_lower = folder.name.lower().replace("_", "-")
        # match scenario prefix (A0, B1, …) và dataset suffix
        if folder.name.upper().startswith(scenario_code.upper()) and \
                dataset.lower() in folder_lower:
            seed_dirs = [d for d in folder.iterdir() if d.is_dir()]
            if seed_dirs:
                m = seed_dirs[0] / "metrics.json"
                if m.exists():
                    return json.loads(m.read_text())
    return None


def build_df() -> pd.DataFrame:
    rows = []
    for model_name, model_dir in MODELS.items():
        for ds in DATASETS:
            for sc in SCENARIOS:
                m = find_metrics(model_dir, ds, sc)
                if m is None:
                    print(f"  [WARN] Missing: {model_name} / {ds} / {sc}")
                    continue
                tg = m.get("test_global", {})
                rows.append({
                    "model":      model_name,
                    "dataset":    ds,
                    "scenario":   sc,
                    "micro_f1":   tg.get("micro_f1", np.nan),
                    "macro_f1":   tg.get("macro_f1", np.nan),
                    "macro_p":    tg.get("macro_precision", np.nan),
                    "macro_r":    tg.get("macro_recall", np.nan),
                    "weighted_f1": tg.get("weighted_f1", np.nan),
                    "hit_at_3":   tg.get("hit_at_3", np.nan),
                    "hit_at_5":   tg.get("hit_at_5", np.nan),
                    "mrr":        tg.get("mrr", np.nan),
                    "map":        tg.get("map", np.nan),
                    "minority_f2": m.get("test_minority_macro_f2", np.nan),
                    "minority_recall": m.get("test_minority_macro_recall", np.nan),
                    "train_sec":  m.get("training_seconds", np.nan),
                    "vram_mb":    m.get("peak_vram_mb", np.nan),
                })
    return pd.DataFrame(rows)


print("📂 Loading metrics …")
df = build_df()
print(f"   Loaded {len(df)} records.\n")


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"   ✅ Saved → {path.name}")
    plt.close(fig)


def bar_label(ax, bars, fmt=".3f", pad=2, fontsize=7, color="white"):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h - pad * 0.01,
            f"{h:{fmt}}",
            ha="center", va="top",
            fontsize=fontsize, color=color, fontweight="bold",
        )


# ─────────────────────────────────────────────
# FIGURE 1 — Grouped bar: 5 scenarios × 3 datasets per model (Macro-F1)
# ─────────────────────────────────────────────
print("📊 Fig 1 – Macro-F1: 5 scenarios × 3 datasets per model")

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
fig.patch.set_facecolor("#0F1117")

x = np.arange(len(SCENARIOS))
width = 0.25
ds_offsets = [-width, 0, width]

for ax, model in zip(axes, ["DistilBERT", "SecureBERT"]):
    ax.set_facecolor("#1A1D27")
    sub = df[df["model"] == model]
    for i, ds in enumerate(DATASETS):
        vals = [sub[(sub["dataset"] == ds) & (sub["scenario"] == sc)]["macro_f1"].values
                for sc in SCENARIOS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        bars = ax.bar(x + ds_offsets[i], vals, width, label=DATASET_LABELS[ds],
                      color=DATASET_COLORS[ds], alpha=0.88, edgecolor="#0F1117", linewidth=0.6)
        bar_label(ax, bars, fontsize=6.5)

    ax.set_title(model, fontsize=14, fontweight="bold", color="white", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS],
                       fontsize=8.5, color="#CCCCCC")
    ax.set_ylim(0.35, 0.90)
    ax.set_ylabel("Macro-F1", fontsize=11, color="white")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=9, framealpha=0.2, labelcolor="white",
                       facecolor="#2A2D3A", edgecolor="#444")
    for t in legend.get_texts():
        t.set_color("white")

fig.suptitle("Macro-F1 by Scenario & Dataset  |  DistilBERT vs SecureBERT",
             fontsize=15, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig1_macro_f1_scenarios_datasets.png")


# ─────────────────────────────────────────────
# FIGURE 2 — Grouped bar: Micro-F1
# ─────────────────────────────────────────────
print("📊 Fig 2 – Micro-F1: 5 scenarios × 3 datasets per model")

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
fig.patch.set_facecolor("#0F1117")

for ax, model in zip(axes, ["DistilBERT", "SecureBERT"]):
    ax.set_facecolor("#1A1D27")
    sub = df[df["model"] == model]
    for i, ds in enumerate(DATASETS):
        vals = [sub[(sub["dataset"] == ds) & (sub["scenario"] == sc)]["micro_f1"].values
                for sc in SCENARIOS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        bars = ax.bar(x + ds_offsets[i], vals, width, label=DATASET_LABELS[ds],
                      color=DATASET_COLORS[ds], alpha=0.88, edgecolor="#0F1117", linewidth=0.6)
        bar_label(ax, bars, fontsize=6.5)

    ax.set_title(model, fontsize=14, fontweight="bold", color="white", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=8.5, color="#CCCCCC")
    ax.set_ylim(0.65, 0.90)
    ax.set_ylabel("Micro-F1", fontsize=11, color="white")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=9, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444")
    for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Micro-F1 by Scenario & Dataset  |  DistilBERT vs SecureBERT",
             fontsize=15, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig2_micro_f1_scenarios_datasets.png")


# ─────────────────────────────────────────────
# FIGURE 3 — Head-to-head bar: DistilBERT vs SecureBERT per scenario per dataset
# ─────────────────────────────────────────────
print("📊 Fig 3 – Head-to-head Macro-F1 per dataset (all scenarios)")

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
fig.patch.set_facecolor("#0F1117")

for ax, ds in zip(axes, DATASETS):
    ax.set_facecolor("#1A1D27")
    sub = df[df["dataset"] == ds]
    x2 = np.arange(len(SCENARIOS))
    w = 0.35
    for j, model in enumerate(["DistilBERT", "SecureBERT"]):
        vals = [sub[(sub["model"] == model) & (sub["scenario"] == sc)]["macro_f1"].values
                for sc in SCENARIOS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = -w/2 if j == 0 else w/2
        bars = ax.bar(x2 + offset, vals, w, label=model,
                      color=MODEL_COLORS[model], alpha=0.88,
                      edgecolor="#0F1117", linewidth=0.7)
        bar_label(ax, bars, fontsize=7)

    ax.set_title(f"Dataset: {DATASET_LABELS[ds]}", fontsize=13, fontweight="bold",
                 color="white", pad=10)
    ax.set_xticks(x2)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=8.5, color="#CCCCCC")
    lo = df[df["dataset"] == ds]["macro_f1"].min() - 0.05
    hi = df[df["dataset"] == ds]["macro_f1"].max() + 0.04
    ax.set_ylim(max(0.3, lo), min(1.0, hi))
    ax.set_ylabel("Macro-F1", fontsize=11, color="white")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=9, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444")
    for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Head-to-Head Macro-F1: DistilBERT vs SecureBERT (per Dataset)",
             fontsize=15, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig3_head2head_macro_f1_per_dataset.png")


# ─────────────────────────────────────────────
# FIGURE 4 — Heatmap: Macro-F1 (model × scenario × dataset)
# ─────────────────────────────────────────────
print("📊 Fig 4 – Heatmap Macro-F1 (all combos)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0F1117")

for ax, model in zip(axes, ["DistilBERT", "SecureBERT"]):
    sub = df[df["model"] == model]
    matrix = pd.DataFrame(index=SCENARIOS, columns=DATASETS, dtype=float)
    for sc in SCENARIOS:
        for ds in DATASETS:
            v = sub[(sub["scenario"] == sc) & (sub["dataset"] == ds)]["macro_f1"].values
            matrix.loc[sc, ds] = v[0] if len(v) else np.nan

    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn",
                   vmin=0.45, vmax=0.85, aspect="auto")
    ax.set_facecolor("#1A1D27")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=11, color="white")
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels(SCENARIOS, fontsize=11, color="white")
    ax.set_title(model, fontsize=13, fontweight="bold", color="white", pad=10)
    ax.tick_params(colors="#AAAAAA")

    for i, sc in enumerate(SCENARIOS):
        for j, ds in enumerate(DATASETS):
            val = matrix.loc[sc, ds]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10.5, fontweight="bold",
                        color="black" if val > 0.62 else "white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label("Macro-F1", color="white", fontsize=9)

fig.suptitle("Heatmap – Macro-F1 across Scenarios & Datasets",
             fontsize=14, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig4_heatmap_macro_f1.png")


# ─────────────────────────────────────────────
# FIGURE 5 — Minority-F2: 2-stage scenarios only
# ─────────────────────────────────────────────
print("📊 Fig 5 – Minority-F2 (2-stage scenarios: B2 & G1)")

two_stage = ["B2", "G1"]
df_min = df[(df["scenario"].isin(two_stage)) & (df["minority_f2"].notna()) & (df["minority_f2"] > 0)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#0F1117")

for ax, ds in zip(axes, DATASETS):
    ax.set_facecolor("#1A1D27")
    sub = df_min[df_min["dataset"] == ds]
    x3 = np.arange(len(two_stage))
    w = 0.35
    for j, model in enumerate(["DistilBERT", "SecureBERT"]):
        vals = [sub[(sub["model"] == model) & (sub["scenario"] == sc)]["minority_f2"].values
                for sc in two_stage]
        vals = [v[0] if len(v) else 0.0 for v in vals]
        offset = -w/2 if j == 0 else w/2
        bars = ax.bar(x3 + offset, vals, w, label=model,
                      color=MODEL_COLORS[model], alpha=0.88,
                      edgecolor="#0F1117", linewidth=0.7)
        bar_label(ax, bars, fontsize=8, pad=1)

    ax.set_title(f"Dataset: {DATASET_LABELS[ds]}", fontsize=13, fontweight="bold",
                 color="white", pad=10)
    ax.set_xticks(x3)
    ax.set_xticklabels(["B2\n(Cyber EDA\n2-stage)", "G1\n(Generic EDA\n2-stage)"],
                       fontsize=9, color="#CCCCCC")
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Minority Macro-F2", fontsize=11, color="white")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=9, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444")
    for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Minority Macro-F₂ (Rare Class Detection)  |  2-Stage Scenarios Only",
             fontsize=14, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig5_minority_f2_2stage.png")


# ─────────────────────────────────────────────
# FIGURE 6 — Augmentation lift: Δ Macro-F1 vs A0 baseline
# ─────────────────────────────────────────────
print("📊 Fig 6 – Augmentation lift (ΔMacro-F1 vs A0 baseline)")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#0F1117")

aug_scenarios = ["B1", "B2", "G0", "G1"]
aug_labels = ["B1\nCyber 1-st", "B2\nCyber 2-st", "G0\nGeneric 1-st", "G1\nGeneric 2-st"]

for ax, ds in zip(axes, DATASETS):
    ax.set_facecolor("#1A1D27")
    x4 = np.arange(len(aug_scenarios))
    w = 0.35

    for j, model in enumerate(["DistilBERT", "SecureBERT"]):
        sub = df[(df["model"] == model) & (df["dataset"] == ds)]
        base = sub[sub["scenario"] == "A0"]["macro_f1"].values
        base = base[0] if len(base) else np.nan
        lifts = []
        for sc in aug_scenarios:
            v = sub[sub["scenario"] == sc]["macro_f1"].values
            lifts.append((v[0] - base) if len(v) else np.nan)

        offset = -w/2 if j == 0 else w/2
        bar_colors = [MODEL_COLORS[model] if (l is not None and not np.isnan(l) and l >= 0)
                      else "#E05050" for l in lifts]
        bars = ax.bar(x4 + offset, lifts, w, label=model,
                      color=MODEL_COLORS[model], alpha=0.88,
                      edgecolor="#0F1117", linewidth=0.7)
        # label
        for bar, v in zip(bars, lifts):
            if v is None or np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.003 if v >= 0 else v - 0.008,
                    f"+{v:.3f}" if v >= 0 else f"{v:.3f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7, color="white", fontweight="bold")

    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
    ax.set_title(f"Dataset: {DATASET_LABELS[ds]}", fontsize=13, fontweight="bold",
                 color="white", pad=10)
    ax.set_xticks(x4)
    ax.set_xticklabels(aug_labels, fontsize=8.5, color="#CCCCCC")
    ax.set_ylabel("ΔMacro-F1 vs A0 Baseline", fontsize=10, color="white")
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=9, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444")
    for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Augmentation Lift (ΔMacro-F1 over A0 Baseline)  |  per Dataset",
             fontsize=14, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig6_augmentation_lift.png")


# ─────────────────────────────────────────────
# FIGURE 7 — Radar chart: multi-metric comparison (TRAM, best scenario each)
# ─────────────────────────────────────────────
print("📊 Fig 7 – Radar chart: multi-metric (TRAM, best macro-f1 scenario)")

# Best scenario trên TRAM: DistilBERT=G0, SecureBERT=B2
best = {
    "DistilBERT": df[(df["model"] == "DistilBERT") & (df["dataset"] == "tram") & (df["scenario"] == "G0")].iloc[0],
    "SecureBERT": df[(df["model"] == "SecureBERT") & (df["dataset"] == "tram") & (df["scenario"] == "B2")].iloc[0],
}
metrics_radar = ["macro_f1", "micro_f1", "weighted_f1", "mrr", "map", "hit_at_3", "hit_at_5"]
metric_labels = ["Macro-F1", "Micro-F1", "Weighted-F1", "MRR", "MAP", "Hit@3", "Hit@5"]

N = len(metrics_radar)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#1A1D27")
ax.spines["polar"].set_color("#444")

for model, color in MODEL_COLORS.items():
    row = best[model]
    vals = [float(row[m]) for m in metrics_radar]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2, color=color, label=model)
    ax.fill(angles, vals, alpha=0.12, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=10.5, color="white")
ax.set_ylim(0.75, 1.0)
ax.yaxis.set_tick_params(labelcolor="#AAAAAA", labelsize=8)
ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(color="#333344", linewidth=0.8)

legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
                   fontsize=10, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444")
for t in legend.get_texts(): t.set_color("white")

ax.set_title("Radar: Multi-Metric Comparison on TRAM\n"
             "(DistilBERT-G0 vs SecureBERT-B2 — Best per Model)",
             fontsize=12, fontweight="bold", color="white", pad=20)
plt.tight_layout()
save(fig, "fig7_radar_multimetric_tram.png")


# ─────────────────────────────────────────────
# FIGURE 8 — Training time & VRAM cost
# ─────────────────────────────────────────────
print("📊 Fig 8 – Computational cost: Training time & VRAM")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0F1117")

# --- 8a: Training time grouped bar ---
ax = axes[0]
ax.set_facecolor("#1A1D27")
x5 = np.arange(len(SCENARIOS))
w = 0.35

for j, model in enumerate(["DistilBERT", "SecureBERT"]):
    # Average across datasets
    vals = []
    for sc in SCENARIOS:
        v = df[(df["model"] == model) & (df["scenario"] == sc)]["train_sec"].mean()
        vals.append(v)
    offset = -w/2 if j == 0 else w/2
    bars = ax.bar(x5 + offset, vals, w, label=model,
                  color=MODEL_COLORS[model], alpha=0.88,
                  edgecolor="#0F1117", linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 15,
                f"{v:.0f}s", ha="center", va="bottom",
                fontsize=7.5, color="white", fontweight="bold")

ax.set_title("Avg Training Time per Scenario", fontsize=12, fontweight="bold", color="white")
ax.set_xticks(x5)
ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=8.5, color="#CCCCCC")
ax.set_ylabel("Seconds", fontsize=11, color="white")
ax.tick_params(colors="#AAAAAA")
ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
ax.spines[:].set_visible(False)
legend = ax.legend(fontsize=9, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444")
for t in legend.get_texts(): t.set_color("white")

# --- 8b: VRAM grouped bar ---
ax = axes[1]
ax.set_facecolor("#1A1D27")

for j, model in enumerate(["DistilBERT", "SecureBERT"]):
    vals = []
    for sc in SCENARIOS:
        v = df[(df["model"] == model) & (df["scenario"] == sc)]["vram_mb"].mean()
        vals.append(v)
    offset = -w/2 if j == 0 else w/2
    bars = ax.bar(x5 + offset, vals, w, label=model,
                  color=MODEL_COLORS[model], alpha=0.88,
                  edgecolor="#0F1117", linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 30,
                f"{v:.0f}", ha="center", va="bottom",
                fontsize=7.5, color="white", fontweight="bold")

ax.set_title("Avg Peak VRAM (MB) per Scenario", fontsize=12, fontweight="bold", color="white")
ax.set_xticks(x5)
ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=8.5, color="#CCCCCC")
ax.set_ylabel("VRAM (MB)", fontsize=11, color="white")
ax.tick_params(colors="#AAAAAA")
ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
ax.spines[:].set_visible(False)
legend = ax.legend(fontsize=9, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444")
for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Computational Cost: Training Time & Peak VRAM",
             fontsize=14, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig8_computational_cost.png")


# ─────────────────────────────────────────────
# FIGURE 9 — Line chart: Macro-F1 across datasets (trend)
# ─────────────────────────────────────────────
print("📊 Fig 9 – Line chart: Macro-F1 across datasets per scenario")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.patch.set_facecolor("#0F1117")

sc_colors = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8"]
ds_x = [0, 1, 2]

for ax, model in zip(axes, ["DistilBERT", "SecureBERT"]):
    ax.set_facecolor("#1A1D27")
    sub = df[df["model"] == model]
    for i, sc in enumerate(SCENARIOS):
        vals = [sub[(sub["scenario"] == sc) & (sub["dataset"] == ds)]["macro_f1"].values
                for ds in DATASETS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.plot(ds_x, vals, "o-", linewidth=2.2, markersize=7,
                color=sc_colors[i], label=sc, alpha=0.9)
        for xi, v in zip(ds_x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.008, f"{v:.3f}", ha="center",
                        fontsize=7, color=sc_colors[i])

    ax.set_title(model, fontsize=13, fontweight="bold", color="white", pad=10)
    ax.set_xticks(ds_x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=11, color="#CCCCCC")
    ax.set_ylabel("Macro-F1", fontsize=11, color="white")
    ax.set_ylim(0.40, 0.90)
    ax.tick_params(colors="#AAAAAA")
    ax.grid(color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(title="Scenario", fontsize=9, framealpha=0.2,
                       facecolor="#2A2D3A", edgecolor="#444",
                       title_fontsize=9)
    legend.get_title().set_color("white")
    for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Macro-F1 Trend across Datasets (CTI-MITRE → Joint → TRAM)",
             fontsize=14, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig9_macro_f1_dataset_trend.png")


# ─────────────────────────────────────────────
# FIGURE 10 — Scatter: Macro-F1 vs Train Time (efficiency frontier)
# ─────────────────────────────────────────────
print("📊 Fig 10 – Scatter: Macro-F1 vs Training time (efficiency)")

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#1A1D27")

for model, mcolor in MODEL_COLORS.items():
    for ds, dcolor in DATASET_COLORS.items():
        sub = df[(df["model"] == model) & (df["dataset"] == ds)]
        for _, row in sub.iterrows():
            ax.scatter(row["train_sec"], row["macro_f1"],
                       s=90, color=dcolor if model == "DistilBERT" else mcolor,
                       marker="o" if model == "DistilBERT" else "^",
                       alpha=0.85, edgecolors="white", linewidth=0.5, zorder=3)
            ax.annotate(f"{row['scenario']}\n{DATASET_LABELS[ds][:4]}",
                        (row["train_sec"], row["macro_f1"]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=5.5, color="#CCCCCC", alpha=0.85)

# Legend
legend_elements = [
    mpatches.Patch(color=MODEL_COLORS["DistilBERT"], label="DistilBERT (○)"),
    mpatches.Patch(color=MODEL_COLORS["SecureBERT"], label="SecureBERT (△)"),
    mpatches.Patch(color=DATASET_COLORS["cti-mitre"], label="CTI-MITRE"),
    mpatches.Patch(color=DATASET_COLORS["joint"],     label="Joint"),
    mpatches.Patch(color=DATASET_COLORS["tram"],      label="TRAM"),
]
legend = ax.legend(handles=legend_elements, fontsize=9, framealpha=0.2,
                   facecolor="#2A2D3A", edgecolor="#444")
for t in legend.get_texts(): t.set_color("white")

ax.set_xlabel("Training Time (seconds)", fontsize=12, color="white")
ax.set_ylabel("Macro-F1", fontsize=12, color="white")
ax.set_title("Efficiency Frontier: Macro-F1 vs Training Time\n"
             "Top-right = best (high accuracy, low cost)",
             fontsize=13, fontweight="bold", color="white")
ax.tick_params(colors="#AAAAAA")
ax.grid(color="#333344", linewidth=0.6, linestyle="--")
ax.spines[:].set_visible(False)
plt.tight_layout()
save(fig, "fig10_efficiency_scatter.png")


# ─────────────────────────────────────────────
# FIGURE 11 — Macro-Recall comparison (same layout as Fig 1)
# ─────────────────────────────────────────────
print("📊 Fig 11 – Macro-Recall: 5 scenarios × 3 datasets per model")

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
fig.patch.set_facecolor("#0F1117")

for ax, model in zip(axes, ["DistilBERT", "SecureBERT"]):
    ax.set_facecolor("#1A1D27")
    sub = df[df["model"] == model]
    for i, ds in enumerate(DATASETS):
        vals = [sub[(sub["dataset"] == ds) & (sub["scenario"] == sc)]["macro_r"].values
                for sc in SCENARIOS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        bars = ax.bar(x + ds_offsets[i], vals, width, label=DATASET_LABELS[ds],
                      color=DATASET_COLORS[ds], alpha=0.88, edgecolor="#0F1117", linewidth=0.6)
        bar_label(ax, bars, fontsize=6.5)

    ax.set_title(model, fontsize=14, fontweight="bold", color="white", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS],
                       fontsize=8.5, color="#CCCCCC")
    ax.set_ylim(0.30, 0.88)
    ax.set_ylabel("Macro-Recall", fontsize=11, color="white")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=9, framealpha=0.2, labelcolor="white",
                       facecolor="#2A2D3A", edgecolor="#444")
    for t in legend.get_texts():
        t.set_color("white")

fig.suptitle("Macro-Recall by Scenario & Dataset  |  DistilBERT vs SecureBERT",
             fontsize=15, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig11_macro_recall_scenarios_datasets.png")


# ─────────────────────────────────────────────
# FIGURE 11b — Head-to-head Macro-Recall per dataset
# ─────────────────────────────────────────────
print("📊 Fig 11b – Head-to-Head Macro-Recall per dataset")

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
fig.patch.set_facecolor("#0F1117")

for ax, ds in zip(axes, DATASETS):
    ax.set_facecolor("#1A1D27")
    sub = df[df["dataset"] == ds]
    x2 = np.arange(len(SCENARIOS))
    w = 0.35
    for j, model in enumerate(["DistilBERT", "SecureBERT"]):
        vals = [sub[(sub["model"] == model) & (sub["scenario"] == sc)]["macro_r"].values
                for sc in SCENARIOS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = -w/2 if j == 0 else w/2
        bars = ax.bar(x2 + offset, vals, w, label=model,
                      color=MODEL_COLORS[model], alpha=0.88,
                      edgecolor="#0F1117", linewidth=0.7)
        bar_label(ax, bars, fontsize=7)

    ax.set_title(f"Dataset: {DATASET_LABELS[ds]}", fontsize=13, fontweight="bold",
                 color="white", pad=10)
    ax.set_xticks(x2)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=8.5, color="#CCCCCC")
    lo = df[df["dataset"] == ds]["macro_r"].min() - 0.05
    hi = df[df["dataset"] == ds]["macro_r"].max() + 0.04
    ax.set_ylim(max(0.3, lo), min(1.0, hi))
    ax.set_ylabel("Macro-Recall", fontsize=11, color="white")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=9, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444")
    for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Head-to-Head Macro-Recall: DistilBERT vs SecureBERT (per Dataset)",
             fontsize=15, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig11b_head2head_macro_recall.png")


# ─────────────────────────────────────────────
# ZERO-F1 LABEL STATISTICS
# ─────────────────────────────────────────────
print("\n📊 Computing zero-F1 label statistics from per_label_global.csv …")

def find_per_label_csv(model_dir: Path, dataset: str, scenario_code: str):
    """Trả về DataFrame per_label_global.csv tương ứng."""
    for folder in model_dir.iterdir():
        if not folder.is_dir():
            continue
        folder_lower = folder.name.lower().replace("_", "-")
        if folder.name.upper().startswith(scenario_code.upper()) and \
                dataset.lower() in folder_lower:
            seed_dirs = [d for d in folder.iterdir() if d.is_dir()]
            if seed_dirs:
                f = seed_dirs[0] / "per_label_global.csv"
                if f.exists():
                    return pd.read_csv(f)
    return None


zero_rows = []
for model_name, model_dir in MODELS.items():
    for ds in DATASETS:
        for sc in SCENARIOS:
            pldf = find_per_label_csv(model_dir, ds, sc)
            if pldf is None:
                print(f"  [WARN] Missing per_label: {model_name}/{ds}/{sc}")
                continue
            total_labels     = len(pldf)
            zero_f1          = (pldf["F1"] == 0.0).sum()
            zero_f1_test_sup = ((pldf["F1"] == 0.0) & (pldf["Test_Support"] > 0)).sum()
            zero_f1_no_test  = ((pldf["F1"] == 0.0) & (pldf["Test_Support"] == 0)).sum()
            zero_rows.append({
                "model":              model_name,
                "dataset":            ds,
                "scenario":           sc,
                "total_labels":       total_labels,
                "zero_f1_total":      int(zero_f1),
                "zero_f1_has_test":   int(zero_f1_test_sup),  # có sample test nhưng không predict được
                "zero_f1_no_test":    int(zero_f1_no_test),   # không có sample test -> trivially 0
                "zero_f1_pct":        round(100 * zero_f1 / total_labels, 1),
                "zero_f1_has_pct":    round(100 * zero_f1_test_sup / total_labels, 1),
            })

df_zero = pd.DataFrame(zero_rows)

# In bảng thống kê ra terminal
print("\n" + "="*80)
print("THONG KE NHAN F1=0 (KHONG DU DOAN DUOC)")
print("="*80)
for ds in DATASETS:
    print(f"\n--- Dataset: {DATASET_LABELS[ds]} ---")
    sub = df_zero[df_zero["dataset"] == ds][
        ["model","scenario","total_labels","zero_f1_total","zero_f1_has_test","zero_f1_no_test","zero_f1_pct"]
    ]
    print(sub.to_string(index=False))

# Lưu CSV
csv_path = FIG_DIR / "zero_f1_label_stats.csv"
df_zero.to_csv(csv_path, index=False)
print(f"\n   Saved CSV -> {csv_path.name}")


# ─────────────────────────────────────────────
# FIGURE 12 — Bar chart: zero-F1 count (total & has-test)
# ─────────────────────────────────────────────
print("📊 Fig 12 – Zero-F1 label count per experiment")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor("#0F1117")

for ax, ds in zip(axes, DATASETS):
    ax.set_facecolor("#1A1D27")
    sub = df_zero[df_zero["dataset"] == ds]
    x6 = np.arange(len(SCENARIOS))
    w = 0.20

    offsets = {"DistilBERT_total": -0.30, "DistilBERT_has": -0.10,
               "SecureBERT_total": +0.10, "SecureBERT_has": +0.30}
    palette = {
        "DistilBERT_total": "#3A7FBF",
        "DistilBERT_has":   "#6AB0F5",
        "SecureBERT_total": "#E07B39",
        "SecureBERT_has":   "#F5B07A",
    }
    label_map = {
        "DistilBERT_total": "DistilBERT – All zero-F1",
        "DistilBERT_has":   "DistilBERT – zero-F1 (has test samples)",
        "SecureBERT_total": "SecureBERT – All zero-F1",
        "SecureBERT_has":   "SecureBERT – zero-F1 (has test samples)",
    }

    for key, offset in offsets.items():
        model, col_suffix = key.split("_", 1)
        col = "zero_f1_total" if col_suffix == "total" else "zero_f1_has_test"
        vals = [sub[(sub["model"] == model) & (sub["scenario"] == sc)][col].values
                for sc in SCENARIOS]
        vals = [int(v[0]) if len(v) else 0 for v in vals]
        bars = ax.bar(x6 + offset, vals, w, label=label_map[key],
                      color=palette[key], alpha=0.88,
                      edgecolor="#0F1117", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                        str(v), ha="center", va="bottom",
                        fontsize=6.5, color="white", fontweight="bold")

    ax.set_title(f"Dataset: {DATASET_LABELS[ds]}", fontsize=13, fontweight="bold",
                 color="white", pad=10)
    ax.set_xticks(x6)
    ax.set_xticklabels(SCENARIOS, fontsize=10, color="#CCCCCC")
    ax.set_ylabel("Number of Labels with F1 = 0", fontsize=10, color="white")
    ax.tick_params(colors="#AAAAAA")
    ax.grid(axis="y", color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=7.5, framealpha=0.2, facecolor="#2A2D3A", edgecolor="#444",
                       loc="upper right")
    for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Labels with F1 = 0 (Undetected) per Experiment\n"
             "Dark = All zero-F1 (incl. no test samples)  |  Light = Zero-F1 with existing test samples",
             fontsize=13, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
save(fig, "fig12_zero_f1_label_count.png")


# ─────────────────────────────────────────────
# FIGURE 13 — Heatmap: zero-F1 label % (has_test only)
# ─────────────────────────────────────────────
print("📊 Fig 13 – Heatmap: % zero-F1 labels (has test samples)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0F1117")

for ax, model in zip(axes, ["DistilBERT", "SecureBERT"]):
    sub = df_zero[df_zero["model"] == model]
    matrix = pd.DataFrame(index=SCENARIOS, columns=DATASETS, dtype=float)
    for sc in SCENARIOS:
        for ds in DATASETS:
            v = sub[(sub["scenario"] == sc) & (sub["dataset"] == ds)]["zero_f1_has_pct"].values
            matrix.loc[sc, ds] = v[0] if len(v) else np.nan

    # Reversed colormap: fewer zeros = better = green
    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn_r",
                   vmin=0, vmax=30, aspect="auto")
    ax.set_facecolor("#1A1D27")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=11, color="white")
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels(SCENARIOS, fontsize=11, color="white")
    ax.set_title(model, fontsize=13, fontweight="bold", color="white", pad=10)
    ax.tick_params(colors="#AAAAAA")

    for i, sc in enumerate(SCENARIOS):
        for j, ds in enumerate(DATASETS):
            val = matrix.loc[sc, ds]
            if not np.isnan(val):
                # also show raw count
                cnt = sub[(sub["scenario"] == sc) & (sub["dataset"] == ds)]["zero_f1_has_test"].values
                cnt = int(cnt[0]) if len(cnt) else 0
                ax.text(j, i, f"{val:.1f}%\n({cnt} labels)",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if val < 15 else "white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label("% zero-F1 labels\n(red = worse)", color="white", fontsize=9)

fig.suptitle("Heatmap – % Labels with F1=0 (Has Test Samples, Undetected)\n"
             "Red = more undetected labels  |  Green = fewer",
             fontsize=13, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
save(fig, "fig13_heatmap_zero_f1_pct.png")


# ─────────────────────────────────────────────
# FIGURE 14 — Line: zero-F1 trend across datasets
# ─────────────────────────────────────────────
print("📊 Fig 14 – Line: zero-F1 (has test) trend across datasets per scenario")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.patch.set_facecolor("#0F1117")

sc_colors_z = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8"]
ds_x = [0, 1, 2]

for ax, model in zip(axes, ["DistilBERT", "SecureBERT"]):
    ax.set_facecolor("#1A1D27")
    sub = df_zero[df_zero["model"] == model]
    for i, sc in enumerate(SCENARIOS):
        vals = [sub[(sub["scenario"] == sc) & (sub["dataset"] == ds)]["zero_f1_has_test"].values
                for ds in DATASETS]
        vals = [int(v[0]) if len(v) else 0 for v in vals]
        ax.plot(ds_x, vals, "o-", linewidth=2.2, markersize=7,
                color=sc_colors_z[i], label=sc, alpha=0.9)
        for xi, v in zip(ds_x, vals):
            ax.text(xi, v + 0.5, str(v), ha="center",
                    fontsize=8, color=sc_colors_z[i], fontweight="bold")

    ax.set_title(model, fontsize=13, fontweight="bold", color="white", pad=10)
    ax.set_xticks(ds_x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=11, color="#CCCCCC")
    ax.set_ylabel("# Labels with F1 = 0 (has test samples)", fontsize=10, color="white")
    ax.tick_params(colors="#AAAAAA")
    ax.grid(color="#333344", linewidth=0.6, linestyle="--")
    ax.spines[:].set_visible(False)
    legend = ax.legend(title="Scenario", fontsize=9, framealpha=0.2,
                       facecolor="#2A2D3A", edgecolor="#444", title_fontsize=9)
    legend.get_title().set_color("white")
    for t in legend.get_texts(): t.set_color("white")

fig.suptitle("Trend: # Undetected Labels (F1=0, has test samples)  CTI-MITRE → Joint → TRAM",
             fontsize=13, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
save(fig, "fig14_zero_f1_trend.png")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"✅ All figures saved to: {FIG_DIR}")
print(f"{'─'*55}")
figs = sorted(FIG_DIR.glob("*.png"))
for f in figs:
    print(f"   📈 {f.name}")
print(f"\n   📄 zero_f1_label_stats.csv")

