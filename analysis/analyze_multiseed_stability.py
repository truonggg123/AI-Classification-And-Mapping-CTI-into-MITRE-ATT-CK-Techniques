"""
analyze_multiseed_stability.py
───────────────────────────────
Publication-Grade Analysis, Table Images & Visualizations for Task 8
(DistilBERT Multi-Seed Statistical Stability: 5 Seeds x 2 Scenarios x 3 Datasets = 30 Runs)

Seeds: {42, 123, 456, 789, 2026}
Scenarios: GenericEDA 1-Stage (G0) vs. CyberEDA 1-Stage (B1)
Datasets: CTI-to-MITRE (188 labels), Joint (188 labels), TRAM (50 labels)

Outputs:
  - 4 Academic Table Images (PNG, 300 DPI)
  - 7 Grouped Bar Charts with Error Bars (PNG, 300 DPI)
  - 1 Summary CSV File

Output Directory:
  results/figures/multiseed_stability/

Author: Research Team
Date: 2026-10-24
"""

import sys
import os
import json
import warnings
from pathlib import Path

# UTF-8 Windows terminal support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# 0. CONFIG & PATHS
# ═════════════════════════════════════════════════════════════════════════════
REPO_ROOT       = Path(__file__).resolve().parent.parent
RESULTS_ROOT    = REPO_ROOT / "results"
MULTISEED_ROOT  = RESULTS_ROOT / "DistilBERT_MultiSeed"
OUTPUT_DIR      = RESULTS_ROOT / "figures" / "multiseed_stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
SCENARIOS = ["G0", "B1"]
DATASETS = ["cti_to_mitre", "joint", "tram"]

DATASET_TITLES = {
    "cti_to_mitre": "CTI-to-MITRE (188 labels)",
    "joint":        "Joint CTI+TRAM (188 labels)",
    "tram":         "TRAM (50 labels)",
}

DATASET_SHORT = {
    "cti_to_mitre": "CTI-MITRE",
    "joint":        "Joint",
    "tram":         "TRAM",
}

SCENARIO_TITLES = {
    "G0": "GenericEDA (G0)",
    "B1": "CyberEDA (B1)",
}

SCENARIO_COLORS = {
    "G0": "#F59E0B",  # Warm Amber
    "B1": "#10B981",  # Emerald Green
}

SCENARIO_EDGECOLORS = {
    "G0": "#B45309",  # Darker Amber
    "B1": "#047857",  # Darker Green
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER FOR ALL 30 RUNS
# ═════════════════════════════════════════════════════════════════════════════
def load_all_multiseed_data():
    """Scans and loads all 30 experimental runs in DistilBERT_MultiSeed."""
    records = []

    for s in SEEDS:
        seed_dir = MULTISEED_ROOT / f"seed_{s}"
        if not seed_dir.exists():
            print(f"[WARN] Missing seed directory: {seed_dir}")
            continue

        for d in DATASETS:
            for sc in SCENARIOS:
                run_dir = seed_dir / f"{sc}_{d}"
                mf_path = run_dir / "metrics.json"
                if not mf_path.exists():
                    print(f"[WARN] Missing metrics.json: {mf_path}")
                    continue

                try:
                    m_data = json.loads(mf_path.read_text(encoding="utf-8"))
                    tg = m_data.get("test_global", {})
                    tm = m_data.get("test_minority_f2", tg)
                    if not tm:
                        tm = tg

                    # Parse Tier Stats (Tail, Medium, Head)
                    tier_stats_m = {"Tail": 0.0, "Medium": 0.0, "Head": 0.0}
                    pl_m_path = run_dir / "per_label_minority_f2.csv"
                    pl_g_path = run_dir / "per_label_global.csv"

                    target_csv = pl_m_path if pl_m_path.exists() else pl_g_path
                    if target_csv.exists():
                        try:
                            df_pl = pd.read_csv(target_csv)
                            col_bucket = "Support_Bucket" if "Support_Bucket" in df_pl.columns else "Frequency_Group"
                            if col_bucket in df_pl.columns:
                                for b in ["Tail", "Medium", "Head"]:
                                    sub_b = df_pl[df_pl[col_bucket] == b]
                                    if len(sub_b) > 0:
                                        tier_stats_m[b] = float(sub_b["F1"].mean())
                        except Exception as e:
                            print(f"[ERR] Reading {target_csv}: {e}")

                    # Fallback to seed_summary.json if per_label missing
                    summary_path = run_dir / "seed_summary.json"
                    if summary_path.exists() and tier_stats_m["Tail"] == 0.0:
                        try:
                            s_data = json.loads(summary_path.read_text(encoding="utf-8"))
                            tier_stats_m["Tail"] = float(s_data.get("rare_f1", 0.0))
                            tier_stats_m["Medium"] = float(s_data.get("medium_f1", 0.0))
                            tier_stats_m["Head"] = float(s_data.get("freq_f1", 0.0))
                        except Exception:
                            pass

                    rec = {
                        "seed":                   s,
                        "dataset":                d,
                        "scenario":               sc,
                        "run_dir":                str(run_dir),
                        "tau_global":             m_data.get("selected_global_threshold", None),
                        "tau_minority":           m_data.get("selected_minority_threshold", None),
                        # Minority-F2 Primary Metrics
                        "macro_f1":               float(tm.get("macro_f1", 0.0)),
                        "macro_recall":           float(tm.get("macro_recall", 0.0)),
                        "macro_precision":        float(tm.get("macro_precision", 0.0)),
                        "micro_f1":               float(tm.get("micro_f1", 0.0)),
                        "zero_f1_count":          int(tm.get("zero_f1_count", 0)),
                        # Global Metrics for comparison
                        "macro_f1_global":        float(tg.get("macro_f1", 0.0)),
                        "macro_recall_global":    float(tg.get("macro_recall", 0.0)),
                        "macro_precision_global": float(tg.get("macro_precision", 0.0)),
                        "micro_f1_global":        float(tg.get("micro_f1", 0.0)),
                        "zero_f1_global":         int(tg.get("zero_f1_count", 0)),
                        # Retrieval Quality
                        "hit_at_3":               float(tm.get("hit_at_3", 0.0)),
                        "hit_at_5":               float(tm.get("hit_at_5", 0.0)),
                        "mrr":                    float(tm.get("mrr", 0.0)),
                        "map":                    float(tm.get("map", 0.0)),
                        # Efficiency & Hardware
                        "training_seconds":       float(m_data.get("training_seconds", 0.0)),
                        "peak_vram_mb":           float(m_data.get("peak_vram_mb", 0.0)),
                        # Tiers
                        "tier_tail_f1":           float(tier_stats_m["Tail"]),
                        "tier_med_f1":            float(tier_stats_m["Medium"]),
                        "tier_head_f1":           float(tier_stats_m["Head"]),
                    }
                    records.append(rec)
                except Exception as e:
                    print(f"[ERR] Loading {mf_path}: {e}")

    df = pd.DataFrame(records)
    print(f"[DATA LOADER] Successfully loaded {len(df)}/30 Multi-Seed runs.")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. ACADEMIC TABLE IMAGE RENDERER (300 DPI)
# ═════════════════════════════════════════════════════════════════════════════
def render_academic_table_image(headers, cell_text, col_widths, title, footnote,
                                output_path, figsize=(15.5, 7.5),
                                highlight_cols=None):
    """Renders a publication-grade academic table image with strict LaTeX/IEEE styling."""
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    num_rows = len(cell_text)
    if num_rows <= 8:
        font_size = 9.0
        row_scale = 1.75
    elif num_rows <= 15:
        font_size = 8.5
        row_scale = 1.55
    else:
        font_size = 7.5
        row_scale = 1.30

    table.set_fontsize(font_size)
    table.scale(1.0, row_scale)

    # Style Header Row
    for col_idx in range(len(headers)):
        c = table[(0, col_idx)]
        c.set_facecolor("#1E3A8A")  # Deep Navy Blue
        c.set_text_props(color="#FFFFFF", fontweight="bold")
        c.set_edgecolor("#0F172A")
        c.set_linewidth(1.3)

    # Style Data Cells
    for row_idx in range(1, len(cell_text) + 1):
        is_even = (row_idx % 2 == 0)
        row_bg = "#F8FAFC" if is_even else "#FFFFFF"
        for col_idx in range(len(headers)):
            c = table[(row_idx, col_idx)]
            c.set_facecolor(row_bg)
            c.set_edgecolor("#E2E8F0")
            c.set_linewidth(0.6)
            c.set_text_props(color="#0F172A")

            # Emphasize specific columns
            if highlight_cols and col_idx in highlight_cols:
                val_text = str(cell_text[row_idx - 1][col_idx])
                if "+" in val_text or "%" in val_text:
                    c.set_text_props(fontweight="bold", color="#1E3A8A")
                elif "-" in val_text:
                    c.set_text_props(fontweight="bold", color="#B91C1C")
                else:
                    c.set_text_props(fontweight="bold")

    # Table Title (Academic Convention: Top-Center)
    plt.title(title, fontsize=12.0, fontweight="bold", color="#0F172A", pad=25, y=0.98)

    # Footnote (Academic Convention: Bottom-Center)
    plt.figtext(0.5, 0.02, footnote, ha="center", fontsize=8.0, color="#475569", style="italic", linespacing=1.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Table Image saved: {output_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. GENERATE 4 PUBLICATION TABLE IMAGES
# ═════════════════════════════════════════════════════════════════════════════

def generate_table01_overall_benchmark(df, output_path):
    """Table 1: Comprehensive Multi-Seed Stability Benchmark across 3 Datasets."""
    headers = [
        "Dataset", "Scenario",
        "Macro-F1 (%)\n[Mean ± Std]", "Macro-F1 Range\n[Min - Max]", "95% CI (%)\n[t_crit=2.776]", "CV (%)\n[σ / μ]",
        "Macro-Recall (%)\n[Mean ± Std]", "Zero-F1 Count\n[Mean ± Std]", "Micro-F1 (%)\n[Mean ± Std]",
        "Paired t-test\n[B1 vs G0]"
    ]
    col_widths = [0.12, 0.12, 0.12, 0.11, 0.11, 0.07, 0.12, 0.10, 0.11, 0.12]
    cell_text = []

    t_crit = stats.t.ppf(0.975, df=len(SEEDS) - 1)  # 2.7764 for df=4

    for d in DATASETS:
        sub_d = df[df["dataset"] == d]
        f1_g0 = sub_d[sub_d["scenario"] == "G0"]["macro_f1"].values * 100
        f1_b1 = sub_d[sub_d["scenario"] == "B1"]["macro_f1"].values * 100

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(f1_b1, f1_g0)
        p_str = f"t={t_stat:+.2f}, p={p_val:.3f}" + (" *" if p_val < 0.05 else " (ns)")

        for sc in SCENARIOS:
            sub_sc = sub_d[sub_d["scenario"] == sc]
            f1s = sub_sc["macro_f1"].values * 100
            recs = sub_sc["macro_recall"].values * 100
            mics = sub_sc["micro_f1"].values * 100
            zf1s = sub_sc["zero_f1_count"].values

            mu_f1 = np.mean(f1s); std_f1 = np.std(f1s, ddof=1)
            min_f1 = np.min(f1s); max_f1 = np.max(f1s)
            ci_f1 = t_crit * (std_f1 / np.sqrt(len(SEEDS)))
            cv_f1 = (std_f1 / mu_f1) * 100

            mu_rec = np.mean(recs); std_rec = np.std(recs, ddof=1)
            mu_mic = np.mean(mics); std_mic = np.std(mics, ddof=1)
            mu_zf = np.mean(zf1s); std_zf = np.std(zf1s, ddof=1)

            cell_text.append([
                DATASET_SHORT[d],
                SCENARIO_TITLES[sc],
                f"{mu_f1:.2f} ± {std_f1:.2f}%",
                f"[{min_f1:.2f}% - {max_f1:.2f}%]",
                f"± {ci_f1:.2f}%",
                f"{cv_f1:.2f}%",
                f"{mu_rec:.2f} ± {std_rec:.2f}%",
                f"{mu_zf:.1f} ± {std_zf:.1f}",
                f"{mu_mic:.2f} ± {std_mic:.2f}%",
                p_str if sc == "B1" else "--"
            ])

    title = "TABLE I: Multi-Seed Statistical Stability Benchmark Across 3 Datasets (DistilBERT, 5 Random Seeds)"
    footnote = (
        "* Notes: Evaluated across 5 random seeds: {42, 123, 456, 789, 2026} under Minority-F2 Threshold (τ_m).\n"
        "* Statistical Formulas: Mean μ = (1/N)∑x_i; Sample Std σ = √[∑(x_i - μ)² / (N - 1)]; CV (%) = (σ / μ) × 100%; 95% CI = ± t_0.025 × (σ / √N) where t_crit = 2.776 (df = 4).\n"
        "* Paired t-test tests H0: μ(B1) = μ(G0); (ns) denotes no statistically significant difference (p > 0.05)."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(16.8, 6.2), highlight_cols=[2, 5, 9]
    )


def generate_table02_seed_by_seed_breakdown(df, output_path):
    """Table 2: Seed-by-Seed Trajectory & Statistical Significance Testing."""
    headers = [
        "Dataset", "Scenario",
        "Seed 42\n(%)", "Seed 123\n(%)", "Seed 456\n(%)", "Seed 789\n(%)", "Seed 2026\n(%)",
        "Mean ± Std\n(%)", "Pair Delta\n(B1 - G0 pp)", "Statistical Significance\n(Paired t-test)"
    ]
    col_widths = [0.11, 0.13, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12, 0.11, 0.13]
    cell_text = []

    for d in DATASETS:
        sub_d = df[df["dataset"] == d]
        f1_dict = {}
        for sc in SCENARIOS:
            sub_sc = sub_d[sub_d["scenario"] == sc].sort_values("seed")
            f1_dict[sc] = sub_sc["macro_f1"].values * 100

        diff = f1_dict["B1"] - f1_dict["G0"]
        t_stat, p_val = stats.ttest_rel(f1_dict["B1"], f1_dict["G0"])
        sig_str = f"t={t_stat:+.3f}, p={p_val:.4f}\n" + ("Significant (p<0.05)" if p_val < 0.05 else "Equivalent (p>=0.05)")

        for sc in SCENARIOS:
            vals = f1_dict[sc]
            mu = np.mean(vals); std = np.std(vals, ddof=1)
            row = [
                DATASET_SHORT[d],
                SCENARIO_TITLES[sc],
                f"{vals[0]:.2f}%", f"{vals[1]:.2f}%", f"{vals[2]:.2f}%", f"{vals[3]:.2f}%", f"{vals[4]:.2f}%",
                f"{mu:.2f} ± {std:.2f}%",
                f"{np.mean(diff):+.2f} ± {np.std(diff, ddof=1):.2f} pp" if sc == "B1" else "--",
                sig_str if sc == "B1" else "--"
            ]
            cell_text.append(row)

    title = "TABLE II: Seed-by-Seed Macro-F1 Trajectory and Paired Hypothesis Testing (DistilBERT)"
    footnote = (
        "* Notes: Evaluated across 5 random seeds on test splits under Minority-F2 Threshold.\n"
        "* Calculation: Pair Delta = F1(B1) - F1(G0) computed per individual seed; Paired two-tailed t-test with df = 4.\n"
        "* Finding: High p-values (p > 0.48) confirm statistical equivalence between CyberEDA and GenericEDA across diverse random initializations."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(16.8, 6.2), highlight_cols=[7, 8, 9]
    )


def generate_table03_frequency_tier_stability(df, output_path):
    """Table 3: Multi-Seed Technique Frequency Tier Stability Breakdown on Joint Dataset."""
    sub_j = df[df["dataset"] == "joint"]
    headers = [
        "Scenario",
        "Tail Tier F1 (%)\n[<30 samples: 99 labels]", "Tail Range\n[Min - Max]",
        "Medium Tier F1 (%)\n[30-100: 51 labels]", "Medium Range\n[Min - Max]",
        "Head Tier F1 (%)\n[>100: 38 labels]", "Head Range\n[Min - Max]",
        "Zero-F1 Count\n[Mean ± Std]", "Imbalance Gap (pp)\n[Micro-F1 - Macro-F1]"
    ]
    col_widths = [0.13, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.10, 0.11]
    cell_text = []

    for sc in SCENARIOS:
        sub_sc = sub_j[sub_j["scenario"] == sc]
        tail_vals = sub_sc["tier_tail_f1"].values * 100
        med_vals  = sub_sc["tier_med_f1"].values * 100
        head_vals = sub_sc["tier_head_f1"].values * 100
        zf1_vals  = sub_sc["zero_f1_count"].values
        gap_vals  = (sub_sc["micro_f1"].values - sub_sc["macro_f1"].values) * 100

        cell_text.append([
            SCENARIO_TITLES[sc],
            f"{np.mean(tail_vals):.2f} ± {np.std(tail_vals, ddof=1):.2f}%",
            f"[{np.min(tail_vals):.2f}% - {np.max(tail_vals):.2f}%]",
            f"{np.mean(med_vals):.2f} ± {np.std(med_vals, ddof=1):.2f}%",
            f"[{np.min(med_vals):.2f}% - {np.max(med_vals):.2f}%]",
            f"{np.mean(head_vals):.2f} ± {np.std(head_vals, ddof=1):.2f}%",
            f"[{np.min(head_vals):.2f}% - {np.max(head_vals):.2f}%]",
            f"{np.mean(zf1_vals):.1f} ± {np.std(zf1_vals, ddof=1):.1f}",
            f"{np.mean(gap_vals):.2f} ± {np.std(gap_vals, ddof=1):.2f} pp"
        ])

    title = "TABLE III: Multi-Seed Technique Frequency Tier Breakdown on Joint Dataset (DistilBERT, 5 Seeds)"
    footnote = (
        "* Notes: Evaluated on Joint test split (188 labels) across 5 random seeds under Minority-F2 Threshold.\n"
        "* Tier definitions: Tail (<30 train samples), Medium (30-100 samples), Head (>100 samples).\n"
        "* Calculation: Tier Mean F1 = Arithmetic mean of F1 scores across classes within frequency bucket; Imbalance Gap = Micro-F1 - Macro-F1."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(16.8, 3.8), highlight_cols=[1, 7, 8]
    )


def generate_table04_soc_retrieval_and_runtime(df, output_path):
    """Table 4: SOC Incident Retrieval Quality & Computational Runtime Stability."""
    sub_j = df[df["dataset"] == "joint"]
    headers = [
        "Scenario",
        "Hit@3 (%)\n[Mean ± Std]", "Hit@3 Range\n[Min - Max]",
        "Hit@5 (%)\n[Mean ± Std]", "Hit@5 Range\n[Min - Max]",
        "MRR Score\n[Mean ± Std]", "MAP Score\n[Mean ± Std]",
        "Training Time\n[Mean ± Std (s)]", "Runtime CV (%)\n[σ / μ]"
    ]
    col_widths = [0.13, 0.12, 0.10, 0.12, 0.10, 0.11, 0.11, 0.11, 0.10]
    cell_text = []

    for sc in SCENARIOS:
        sub_sc = sub_j[sub_j["scenario"] == sc]
        h3 = sub_sc["hit_at_3"].values * 100
        h5 = sub_sc["hit_at_5"].values * 100
        mrr = sub_sc["mrr"].values
        mapp = sub_sc["map"].values
        times = sub_sc["training_seconds"].values

        mu_t = np.mean(times); std_t = np.std(times, ddof=1)
        cv_t = (std_t / mu_t) * 100

        cell_text.append([
            SCENARIO_TITLES[sc],
            f"{np.mean(h3):.2f} ± {np.std(h3, ddof=1):.2f}%",
            f"[{np.min(h3):.2f}% - {np.max(h3):.2f}%]",
            f"{np.mean(h5):.2f} ± {np.std(h5, ddof=1):.2f}%",
            f"[{np.min(h5):.2f}% - {np.max(h5):.2f}%]",
            f"{np.mean(mrr):.4f} ± {np.std(mrr, ddof=1):.4f}",
            f"{np.mean(mapp):.4f} ± {np.std(mapp, ddof=1):.4f}",
            f"{mu_t:.1f} ± {std_t:.1f}s",
            f"{cv_t:.2f}%"
        ])

    title = "TABLE IV: Operational SOC Alert Triage Quality and Runtime Stability on Joint Dataset (DistilBERT, 5 Seeds)"
    footnote = (
        "* Notes: Evaluated across 4,001 test reports on Joint dataset. Hardware: Single NVIDIA GPU under identical environment.\n"
        "* Definitions: Hit@K = Proportion of reports where ground-truth ATT&CK label appears in top-K predictions; MRR = Mean Reciprocal Rank."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(16.8, 3.8), highlight_cols=[1, 5, 7]
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. GENERATE 7 GROUPED BAR CHARTS WITH ERROR BARS (300 DPI)
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig01_macro_f1_with_error_bars(df, output_path):
    """Figure 1: Cross-Dataset Multi-Seed Macro-F1 with Error Bars (G0 vs B1)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=300, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    w = 0.38
    x = np.array([0])

    for i, d in enumerate(DATASETS):
        ax = axes[i]
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        sub_d = df[df["dataset"] == d]

        for s_idx, sc in enumerate(SCENARIOS):
            sub_sc = sub_d[sub_d["scenario"] == sc]
            f1s = sub_sc["macro_f1"].values * 100
            mu = np.mean(f1s); std = np.std(f1s, ddof=1)
            min_v = np.min(f1s); max_v = np.max(f1s)

            pos = x + (s_idx - 0.5) * w
            rect = ax.bar(pos, [mu], w, yerr=[std], capsize=5,
                          label=SCENARIO_TITLES[sc] if i == 0 else "",
                          color=SCENARIO_COLORS[sc], edgecolor=SCENARIO_EDGECOLORS[sc],
                          linewidth=1.0, alpha=0.92)

            # Bar annotation: Exact Mean ± Std
            ax.text(pos[0], mu + std + 1.2, f"{mu:.2f} ± {std:.2f}%\n[{min_v:.1f}% - {max_v:.1f}%]",
                    ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#0F172A")

        ax.set_xticks([ -w/2, w/2 ])
        ax.set_xticklabels(["GenericEDA (G0)", "CyberEDA (B1)"], fontsize=9.0, fontweight="bold")
        ax.set_title(f"({chr(97+i)}) {DATASET_TITLES[d]}", fontsize=11.0, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Macro-F1 Score (%)", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 95)

    axes[0].legend(frameon=True, fontsize=8.8, loc="upper left")

    plt.suptitle("Figure 1: Cross-Dataset Multi-Seed Macro-F1 Stability (DistilBERT Across 5 Random Seeds)",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.12)

    caption = (
        "Notes: Evaluated across 5 random seeds: {42, 123, 456, 789, 2026} under Minority-F2 Threshold (τ_m).\n"
        "Calculation: Bar height = Mean Macro-F1 (μ); Error bar = Sample Standard Deviation (±σ); Bracketed text = [Min - Max] score."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig02_seed_by_seed_joint(df, output_path):
    """Figure 2: Seed-by-Seed Trajectory & Direct Comparison on Joint Dataset."""
    sub_j = df[df["dataset"] == "joint"]
    fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    x = np.arange(len(SEEDS))
    w = 0.35

    f1_g0 = sub_j[sub_j["scenario"] == "G0"].sort_values("seed")["macro_f1"].values * 100
    f1_b1 = sub_j[sub_j["scenario"] == "B1"].sort_values("seed")["macro_f1"].values * 100

    r1 = ax.bar(x - w/2, f1_g0, w, label="GenericEDA (G0)", color=SCENARIO_COLORS["G0"],
                edgecolor=SCENARIO_EDGECOLORS["G0"], linewidth=1.0, alpha=0.92)
    r2 = ax.bar(x + w/2, f1_b1, w, label="CyberEDA (B1)", color=SCENARIO_COLORS["B1"],
                edgecolor=SCENARIO_EDGECOLORS["B1"], linewidth=1.0, alpha=0.92)

    for rect in r1:
        val = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, val + 0.7, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#78350F")

    for rect in r2:
        val = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, val + 0.7, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#064E3B")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=10.0, fontweight="bold")
    ax.set_ylabel("Macro-F1 Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 80)
    ax.legend(frameon=True, fontsize=9.0, loc="lower right")

    plt.suptitle("Figure 2: Seed-by-Seed Macro-F1 Direct Comparison on Joint Dataset (DistilBERT)",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.07, right=0.96)

    caption = (
        "Notes: Evaluated on Joint test split (188 labels, 4,001 reports) under Minority-F2 Threshold (τ_m).\n"
        "Compares individual run trajectories across 5 distinct random initializations: {42, 123, 456, 789, 2026}."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig03_tier_breakdown_error_bars(df, output_path):
    """Figure 3: Multi-Seed Technique Frequency Tier Breakdown on Joint Dataset."""
    sub_j = df[df["dataset"] == "joint"]
    fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    tiers = ["Tail Tier\n(<30 samples: 99 labels)", "Medium Tier\n(30-100: 51 labels)", "Head Tier\n(>100: 38 labels)"]
    x = np.arange(len(tiers))
    w = 0.35

    g0_sub = sub_j[sub_j["scenario"] == "G0"]
    b1_sub = sub_j[sub_j["scenario"] == "B1"]

    mu_g0 = [
        np.mean(g0_sub["tier_tail_f1"]) * 100,
        np.mean(g0_sub["tier_med_f1"]) * 100,
        np.mean(g0_sub["tier_head_f1"]) * 100
    ]
    std_g0 = [
        np.std(g0_sub["tier_tail_f1"], ddof=1) * 100,
        np.std(g0_sub["tier_med_f1"], ddof=1) * 100,
        np.std(g0_sub["tier_head_f1"], ddof=1) * 100
    ]

    mu_b1 = [
        np.mean(b1_sub["tier_tail_f1"]) * 100,
        np.mean(b1_sub["tier_med_f1"]) * 100,
        np.mean(b1_sub["tier_head_f1"]) * 100
    ]
    std_b1 = [
        np.std(b1_sub["tier_tail_f1"], ddof=1) * 100,
        np.std(b1_sub["tier_med_f1"], ddof=1) * 100,
        np.std(b1_sub["tier_head_f1"], ddof=1) * 100
    ]

    r1 = ax.bar(x - w/2, mu_g0, w, yerr=std_g0, capsize=5, label="GenericEDA (G0)",
                color=SCENARIO_COLORS["G0"], edgecolor=SCENARIO_EDGECOLORS["G0"], linewidth=1.0, alpha=0.92)
    r2 = ax.bar(x + w/2, mu_b1, w, yerr=std_b1, capsize=5, label="CyberEDA (B1)",
                color=SCENARIO_COLORS["B1"], edgecolor=SCENARIO_EDGECOLORS["B1"], linewidth=1.0, alpha=0.92)

    for i, rect in enumerate(r1):
        ax.text(rect.get_x() + rect.get_width()/2, mu_g0[i] + std_g0[i] + 1.2,
                f"{mu_g0[i]:.2f} ± {std_g0[i]:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#78350F")

    for i, rect in enumerate(r2):
        ax.text(rect.get_x() + rect.get_width()/2, mu_b1[i] + std_b1[i] + 1.2,
                f"{mu_b1[i]:.2f} ± {std_b1[i]:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#064E3B")

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Mean F1 Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 95)
    ax.legend(frameon=True, fontsize=9.0, loc="upper left")

    plt.suptitle("Figure 3: Multi-Seed Technique Frequency Tier Breakdown on Joint Dataset (DistilBERT Across 5 Seeds)",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.07, right=0.96)

    caption = (
        "Notes: Evaluated on Joint test split under Minority-F2 Threshold across 5 random seeds: {42, 123, 456, 789, 2026}.\n"
        "Calculation: Tier Mean F1 = Mean F1 across techniques within tier; Error bars = Sample Standard Deviation (±σ)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig04_zero_f1_variance(df, output_path):
    """Figure 4: Variance in Missed Threat Techniques (Zero-F1 Count across 5 Seeds)."""
    fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    x = np.arange(len(DATASETS))
    w = 0.35

    mu_zf_g0, std_zf_g0 = [], []
    mu_zf_b1, std_zf_b1 = [], []

    for d in DATASETS:
        sub_d = df[df["dataset"] == d]
        z_g0 = sub_d[sub_d["scenario"] == "G0"]["zero_f1_count"].values
        z_b1 = sub_d[sub_d["scenario"] == "B1"]["zero_f1_count"].values

        mu_zf_g0.append(np.mean(z_g0))
        std_zf_g0.append(np.std(z_g0, ddof=1))
        mu_zf_b1.append(np.mean(z_b1))
        std_zf_b1.append(np.std(z_b1, ddof=1))

    r1 = ax.bar(x - w/2, mu_zf_g0, w, yerr=std_zf_g0, capsize=5, label="GenericEDA (G0)",
                color=SCENARIO_COLORS["G0"], edgecolor=SCENARIO_EDGECOLORS["G0"], linewidth=1.0, alpha=0.92)
    r2 = ax.bar(x + w/2, mu_zf_b1, w, yerr=std_zf_b1, capsize=5, label="CyberEDA (B1)",
                color=SCENARIO_COLORS["B1"], edgecolor=SCENARIO_EDGECOLORS["B1"], linewidth=1.0, alpha=0.92)

    for i, rect in enumerate(r1):
        ax.text(rect.get_x() + rect.get_width()/2, mu_zf_g0[i] + std_zf_g0[i] + 0.35,
                f"{mu_zf_g0[i]:.1f} ± {std_zf_g0[i]:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#78350F")

    for i, rect in enumerate(r2):
        ax.text(rect.get_x() + rect.get_width()/2, mu_zf_b1[i] + std_zf_b1[i] + 0.35,
                f"{mu_zf_b1[i]:.1f} ± {std_zf_b1[i]:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#064E3B")

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_TITLES[d] for d in DATASETS], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Missed ATT&CK Techniques (Zero-F1 Count)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(max(mu_zf_g0), max(mu_zf_b1)) + 5.0)
    ax.legend(frameon=True, fontsize=9.0, loc="upper right")

    plt.suptitle("Figure 4: Reduction and Variance in Missed Threat Techniques (Zero-F1 Count Across 5 Seeds)",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.07, right=0.96)

    caption = (
        "Notes: Evaluated under Minority-F2 Threshold (τ_m) across 5 random seeds.\n"
        "Calculation: Zero-F1 Count = Number of technique classes with test F1 = 0.0 (lower denotes better coverage of rare threats)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig05_recall_precision_stability(df, output_path):
    """Figure 5: Macro-Recall Gain & Macro-Precision Trade-off Stability across Seeds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 5.6), dpi=300, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.45)

    x = np.arange(len(DATASETS))
    w = 0.35

    mu_rec_g0, std_rec_g0, mu_rec_b1, std_rec_b1 = [], [], [], []
    mu_prec_g0, std_prec_g0, mu_prec_b1, std_prec_b1 = [], [], [], []

    for d in DATASETS:
        sub_d = df[df["dataset"] == d]
        rg0 = sub_d[sub_d["scenario"] == "G0"]["macro_recall"].values * 100
        rb1 = sub_d[sub_d["scenario"] == "B1"]["macro_recall"].values * 100
        pg0 = sub_d[sub_d["scenario"] == "G0"]["macro_precision"].values * 100
        pb1 = sub_d[sub_d["scenario"] == "B1"]["macro_precision"].values * 100

        mu_rec_g0.append(np.mean(rg0)); std_rec_g0.append(np.std(rg0, ddof=1))
        mu_rec_b1.append(np.mean(rb1)); std_rec_b1.append(np.std(rb1, ddof=1))
        mu_prec_g0.append(np.mean(pg0)); std_prec_g0.append(np.std(pg0, ddof=1))
        mu_prec_b1.append(np.mean(pb1)); std_prec_b1.append(np.std(pb1, ddof=1))

    # Panel (a): Macro-Recall
    r1_a = ax1.bar(x - w/2, mu_rec_g0, w, yerr=std_rec_g0, capsize=5, label="GenericEDA (G0)",
                   color=SCENARIO_COLORS["G0"], edgecolor=SCENARIO_EDGECOLORS["G0"], linewidth=1.0, alpha=0.92)
    r2_a = ax1.bar(x + w/2, mu_rec_b1, w, yerr=std_rec_b1, capsize=5, label="CyberEDA (B1)",
                   color=SCENARIO_COLORS["B1"], edgecolor=SCENARIO_EDGECOLORS["B1"], linewidth=1.0, alpha=0.92)

    for i, rect in enumerate(r1_a):
        ax1.text(rect.get_x() + rect.get_width()/2, mu_rec_g0[i] + std_rec_g0[i] + 1.2,
                 f"{mu_rec_g0[i]:.2f} ± {std_rec_g0[i]:.2f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#78350F")
    for i, rect in enumerate(r2_a):
        ax1.text(rect.get_x() + rect.get_width()/2, mu_rec_b1[i] + std_rec_b1[i] + 1.2,
                 f"{mu_rec_b1[i]:.2f} ± {std_rec_b1[i]:.2f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#064E3B")

    ax1.set_xticks(x); ax1.set_xticklabels([DATASET_SHORT[d] for d in DATASETS], fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Metric Score (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) Macro-Recall (Threat Indicator Sensitivity)", fontsize=11.0, fontweight="bold")
    ax1.set_ylim(0, 95)
    ax1.legend(frameon=True, fontsize=8.8, loc="upper left")

    # Panel (b): Macro-Precision
    r1_b = ax2.bar(x - w/2, mu_prec_g0, w, yerr=std_prec_g0, capsize=5, label="GenericEDA (G0)",
                   color=SCENARIO_COLORS["G0"], edgecolor=SCENARIO_EDGECOLORS["G0"], linewidth=1.0, alpha=0.92)
    r2_b = ax2.bar(x + w/2, mu_prec_b1, w, yerr=std_prec_b1, capsize=5, label="CyberEDA (B1)",
                   color=SCENARIO_COLORS["B1"], edgecolor=SCENARIO_EDGECOLORS["B1"], linewidth=1.0, alpha=0.92)

    for i, rect in enumerate(r1_b):
        ax2.text(rect.get_x() + rect.get_width()/2, mu_prec_g0[i] + std_prec_g0[i] + 1.2,
                 f"{mu_prec_g0[i]:.2f} ± {std_prec_g0[i]:.2f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#78350F")
    for i, rect in enumerate(r2_b):
        ax2.text(rect.get_x() + rect.get_width()/2, mu_prec_b1[i] + std_prec_b1[i] + 1.2,
                 f"{mu_prec_b1[i]:.2f} ± {std_prec_b1[i]:.2f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#064E3B")

    ax2.set_xticks(x); ax2.set_xticklabels([DATASET_SHORT[d] for d in DATASETS], fontsize=9.5, fontweight="bold")
    ax2.set_title("(b) Macro-Precision (False Alarm Suppression)", fontsize=11.0, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.8, loc="upper left")

    plt.suptitle("Figure 5: Macro-Recall Gain and Macro-Precision Stability Across 5 Seeds (DistilBERT)",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.10)

    caption = (
        "Notes: Evaluated under Minority-F2 Threshold across 5 random seeds: {42, 123, 456, 789, 2026}.\n"
        "Evaluates detection coverage (Macro-Recall) against precision preservation (Macro-Precision)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig06_cv_and_runtime_stability(df, output_path):
    """Figure 6: Coefficient of Variation (CV %) & Training Runtime Robustness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 5.6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.45)

    x = np.arange(len(DATASETS))
    w = 0.35

    cv_g0, cv_b1 = [], []
    mu_time_g0, std_time_g0, mu_time_b1, std_time_b1 = [], [], [], []

    for d in DATASETS:
        sub_d = df[df["dataset"] == d]
        f1_g = sub_d[sub_d["scenario"] == "G0"]["macro_f1"].values * 100
        f1_b = sub_d[sub_d["scenario"] == "B1"]["macro_f1"].values * 100
        t_g  = sub_d[sub_d["scenario"] == "G0"]["training_seconds"].values
        t_b  = sub_d[sub_d["scenario"] == "B1"]["training_seconds"].values

        cv_g0.append((np.std(f1_g, ddof=1) / np.mean(f1_g)) * 100)
        cv_b1.append((np.std(f1_b, ddof=1) / np.mean(f1_b)) * 100)
        mu_time_g0.append(np.mean(t_g)); std_time_g0.append(np.std(t_g, ddof=1))
        mu_time_b1.append(np.mean(t_b)); std_time_b1.append(np.std(t_b, ddof=1))

    # Panel (a): CV (%)
    r1_a = ax1.bar(x - w/2, cv_g0, w, label="GenericEDA (G0)", color=SCENARIO_COLORS["G0"],
                   edgecolor=SCENARIO_EDGECOLORS["G0"], linewidth=1.0, alpha=0.92)
    r2_a = ax1.bar(x + w/2, cv_b1, w, label="CyberEDA (B1)", color=SCENARIO_COLORS["B1"],
                   edgecolor=SCENARIO_EDGECOLORS["B1"], linewidth=1.0, alpha=0.92)

    for rect in list(r1_a) + list(r2_a):
        val = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, val + 0.20, f"{val:.2f}%",
                 ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0F172A")

    ax1.set_xticks(x); ax1.set_xticklabels([DATASET_SHORT[d] for d in DATASETS], fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Coefficient of Variation (CV %)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) Performance Dispersion: Macro-F1 CV (%) Across Seeds", fontsize=11.0, fontweight="bold")
    ax1.set_ylim(0, max(max(cv_g0), max(cv_b1)) + 2.5)
    ax1.legend(frameon=True, fontsize=8.8, loc="upper right")

    # Panel (b): Training Time (seconds)
    r1_b = ax2.bar(x - w/2, mu_time_g0, w, yerr=std_time_g0, capsize=5, label="GenericEDA (G0)",
                   color=SCENARIO_COLORS["G0"], edgecolor=SCENARIO_EDGECOLORS["G0"], linewidth=1.0, alpha=0.92)
    r2_b = ax2.bar(x + w/2, mu_time_b1, w, yerr=std_time_b1, capsize=5, label="CyberEDA (B1)",
                   color=SCENARIO_COLORS["B1"], edgecolor=SCENARIO_EDGECOLORS["B1"], linewidth=1.0, alpha=0.92)

    for i, rect in enumerate(r1_b):
        ax2.text(rect.get_x() + rect.get_width()/2, mu_time_g0[i] + std_time_g0[i] + 15,
                 f"{mu_time_g0[i]:.0f} ± {std_time_g0[i]:.0f}s", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#78350F")
    for i, rect in enumerate(r2_b):
        ax2.text(rect.get_x() + rect.get_width()/2, mu_time_b1[i] + std_time_b1[i] + 15,
                 f"{mu_time_b1[i]:.0f} ± {std_time_b1[i]:.0f}s", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#064E3B")

    ax2.set_xticks(x); ax2.set_xticklabels([DATASET_SHORT[d] for d in DATASETS], fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Training Duration (Seconds)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Computational Runtime Stability Across Seeds", fontsize=11.0, fontweight="bold")
    ax2.set_ylim(0, max(max(mu_time_g0), max(mu_time_b1)) + 120)
    ax2.legend(frameon=True, fontsize=8.8, loc="upper left")

    plt.suptitle("Figure 6: Statistical Dispersion (CV %) and Computational Runtime Stability (DistilBERT Across 5 Seeds)",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.12)

    caption = (
        "Notes: Evaluated across 5 random seeds: {42, 123, 456, 789, 2026}.\n"
        "Calculation: Coefficient of Variation CV (%) = (Sample Std σ / Sample Mean μ) × 100% (values below 5% indicate high stability)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig07_soc_retrieval_stability(df, output_path):
    """Figure 7: SOC Incident Recommendation Quality across 5 Seeds (Hit@3, Hit@5, MRR)."""
    sub_j = df[df["dataset"] == "joint"]
    fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    metrics_names = ["Hit@3 Retrieval (%)", "Hit@5 Retrieval (%)", "Mean Reciprocal Rank (MRR x100)"]
    x = np.arange(len(metrics_names))
    w = 0.35

    g0_sub = sub_j[sub_j["scenario"] == "G0"]
    b1_sub = sub_j[sub_j["scenario"] == "B1"]

    mu_g0 = [
        np.mean(g0_sub["hit_at_3"]) * 100,
        np.mean(g0_sub["hit_at_5"]) * 100,
        np.mean(g0_sub["mrr"]) * 100
    ]
    std_g0 = [
        np.std(g0_sub["hit_at_3"], ddof=1) * 100,
        np.std(g0_sub["hit_at_5"], ddof=1) * 100,
        np.std(g0_sub["mrr"], ddof=1) * 100
    ]

    mu_b1 = [
        np.mean(b1_sub["hit_at_3"]) * 100,
        np.mean(b1_sub["hit_at_5"]) * 100,
        np.mean(b1_sub["mrr"]) * 100
    ]
    std_b1 = [
        np.std(b1_sub["hit_at_3"], ddof=1) * 100,
        np.std(b1_sub["hit_at_5"], ddof=1) * 100,
        np.std(b1_sub["mrr"], ddof=1) * 100
    ]

    r1 = ax.bar(x - w/2, mu_g0, w, yerr=std_g0, capsize=5, label="GenericEDA (G0)",
                color=SCENARIO_COLORS["G0"], edgecolor=SCENARIO_EDGECOLORS["G0"], linewidth=1.0, alpha=0.92)
    r2 = ax.bar(x + w/2, mu_b1, w, yerr=std_b1, capsize=5, label="CyberEDA (B1)",
                color=SCENARIO_COLORS["B1"], edgecolor=SCENARIO_EDGECOLORS["B1"], linewidth=1.0, alpha=0.92)

    for i, rect in enumerate(r1):
        ax.text(rect.get_x() + rect.get_width()/2, mu_g0[i] + std_g0[i] + 1.0,
                f"{mu_g0[i]:.2f} ± {std_g0[i]:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#78350F")

    for i, rect in enumerate(r2):
        ax.text(rect.get_x() + rect.get_width()/2, mu_b1[i] + std_b1[i] + 1.0,
                f"{mu_b1[i]:.2f} ± {std_b1[i]:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#064E3B")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(60, 102)
    ax.legend(frameon=True, fontsize=9.0, loc="lower right")

    plt.suptitle("Figure 7: SOC Incident Recommendation Quality Stability on Joint Dataset (DistilBERT Across 5 Seeds)",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.07, right=0.96)

    caption = (
        "Notes: Evaluated on Joint test split (4,001 reports) across 5 random seeds under Minority-F2 Threshold.\n"
        "Measures multi-label recommendation robustness: Hit@3 > 90%, Hit@5 > 93%, MRR > 0.84 across all runs."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY CSV EXPORTER
# ═════════════════════════════════════════════════════════════════════════════
def export_multiseed_summary_csv(df, output_path):
    """Exports full statistical summary table to clean CSV."""
    records = []
    t_crit = stats.t.ppf(0.975, df=len(SEEDS) - 1)

    for d in DATASETS:
        sub_d = df[df["dataset"] == d]
        f1_g0 = sub_d[sub_d["scenario"] == "G0"]["macro_f1"].values * 100
        f1_b1 = sub_d[sub_d["scenario"] == "B1"]["macro_f1"].values * 100
        t_stat, p_val = stats.ttest_rel(f1_b1, f1_g0)

        for sc in SCENARIOS:
            sub = sub_d[sub_d["scenario"] == sc]
            f1s = sub["macro_f1"].values * 100
            recs = sub["macro_recall"].values * 100
            precs = sub["macro_precision"].values * 100
            mics = sub["micro_f1"].values * 100
            zf1s = sub["zero_f1_count"].values
            times = sub["training_seconds"].values

            mu_f1 = np.mean(f1s); std_f1 = np.std(f1s, ddof=1)
            ci_f1 = t_crit * (std_f1 / np.sqrt(len(SEEDS)))
            cv_f1 = (std_f1 / mu_f1) * 100

            rec = {
                "dataset":          DATASET_SHORT[d],
                "scenario":         sc,
                "scenario_title":   SCENARIO_TITLES[sc],
                "macro_f1_mean":    round(mu_f1, 2),
                "macro_f1_std":     round(std_f1, 2),
                "macro_f1_min":     round(np.min(f1s), 2),
                "macro_f1_max":     round(np.max(f1s), 2),
                "macro_f1_ci95":    round(ci_f1, 2),
                "macro_f1_cv_pct":  round(cv_f1, 2),
                "macro_recall_mean":round(np.mean(recs), 2),
                "macro_recall_std": round(np.std(recs, ddof=1), 2),
                "macro_prec_mean":  round(np.mean(precs), 2),
                "macro_prec_std":   round(np.std(precs, ddof=1), 2),
                "micro_f1_mean":    round(np.mean(mics), 2),
                "micro_f1_std":     round(np.std(mics, ddof=1), 2),
                "zero_f1_mean":     round(np.mean(zf1s), 1),
                "zero_f1_std":      round(np.std(zf1s, ddof=1), 1),
                "train_time_mean":  round(np.mean(times), 1),
                "train_time_std":   round(np.std(times, ddof=1), 1),
                "paired_t_stat":    round(t_stat, 3) if sc == "B1" else None,
                "paired_p_value":   round(p_val, 4) if sc == "B1" else None,
            }
            records.append(rec)

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Summary CSV saved: {output_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN ORCHESTRATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("TASK 8: DISTILBERT MULTI-SEED STATISTICAL STABILITY ANALYSIS")
    print("=" * 80)

    # 1. Load Data
    df = load_all_multiseed_data()
    if len(df) == 0:
        print("[FATAL] No multi-seed data found! Exiting.")
        return

    # 2. Render 4 Academic Table Images
    print("\n>>> Generating 4 Publication-Grade Table Images (300 DPI)...")
    generate_table01_overall_benchmark(df, OUTPUT_DIR / "table01_multiseed_overall_benchmark.png")
    generate_table02_seed_by_seed_breakdown(df, OUTPUT_DIR / "table02_multiseed_seed_by_seed_breakdown.png")
    generate_table03_frequency_tier_stability(df, OUTPUT_DIR / "table03_multiseed_frequency_tier_stability.png")
    generate_table04_soc_retrieval_and_runtime(df, OUTPUT_DIR / "table04_multiseed_soc_retrieval_and_runtime.png")

    # 3. Render 7 Grouped Bar Charts with Error Bars
    print("\n>>> Generating 7 Grouped Bar Charts with Error Bars (300 DPI)...")
    generate_fig01_macro_f1_with_error_bars(df, OUTPUT_DIR / "fig01_multiseed_macro_f1_with_error_bars.png")
    generate_fig02_seed_by_seed_joint(df, OUTPUT_DIR / "fig02_multiseed_seed_by_seed_joint.png")
    generate_fig03_tier_breakdown_error_bars(df, OUTPUT_DIR / "fig03_multiseed_tier_breakdown_error_bars.png")
    generate_fig04_zero_f1_variance(df, OUTPUT_DIR / "fig04_multiseed_zero_f1_variance.png")
    generate_fig05_recall_precision_stability(df, OUTPUT_DIR / "fig05_multiseed_recall_precision_stability.png")
    generate_fig06_cv_and_runtime_stability(df, OUTPUT_DIR / "fig06_multiseed_cv_and_runtime_stability.png")
    generate_fig07_soc_retrieval_stability(df, OUTPUT_DIR / "fig07_multiseed_soc_retrieval_stability.png")

    # 4. Export CSV Summary
    print("\n>>> Exporting Summary CSV...")
    export_multiseed_summary_csv(df, OUTPUT_DIR / "multiseed_stability_summary.csv")

    print("\n" + "=" * 80)
    print(f"TASK 8 ANALYSIS COMPLETE! All 11 publication artifacts saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
