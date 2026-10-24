"""
analyze_main_experiments.py
───────────────────────────
Publication-Grade Analysis, Table Images & Visualizations for Task 7
(7 Deep Learning Architectures x 3 Datasets x 5 Augmentation Scenarios = 105 Runs)

Key Academic Design Principles:
1. Prominent Top Figure Titles & Clean Bottom Notes:
   - Chart titles are placed clearly and prominently at the TOP of each figure.
   - Explanatory footnotes are placed strictly at the BOTTOM, specifying:
     * Datasets being evaluated.
     * Threshold mode (Global Threshold vs. Minority-F2 Threshold).
     * Calculation formulas ONLY for newly derived metrics not present in metrics.json
       (e.g., Imbalance Gap, Performance Margin, Training Overhead, Tier Mean F1).
     * No subjective narrative conclusions (user performs independent analysis).
2. Symmetrical Benchmarking Across All 7 Architectures:
   - TextCNN is evaluated alongside all 6 Transformer architectures across all figures and tables.
   - Evaluated under identical experimental settings to ensure a fair and consistent baseline comparison.
3. Core Figure Progression:
   - Figure 1: Comprehensive Scenario Benchmark (5 Scenarios x 7 Models x 3 Datasets) [Min-F2]
   - Figure 2: ATT&CK Technique Frequency Tier Breakdown (Tail vs Med vs Head) [A0 vs G0 vs B1]
   - Figure 3: Class Imbalance Evidence: Contraction of Macro vs Micro-F1 Disparity [A0 vs G0 vs B1]
   - Figure 4: Neutral Head-to-Head: CyberEDA vs GenericEDA Across Datasets (1-Stage & 2-Stage)
   - Figure 5: Reduction in Missed Threat Techniques (Zero-F1: A0 vs G0 vs B1 under Glob vs Min-F2)
   - Figure 6: Macro-Recall Gain & Precision Trade-off (A0 vs G0 vs B1 under Glob vs Min-F2)
   - Figure 7: Reference Benchmark under Standard Global Threshold (5 Scenarios x 7 Models)
   - Figure 8: Computational Cost & Overhead: GenericEDA vs CyberEDA (A0, G0, B1, G1, B2)
   - Figure 9: Operational Retrieval Quality for SOC Incident Response (Hit@3, Hit@5, MRR)
4. 7 Publication-Grade Academic Table Images (PNG, 300 DPI) with side-by-side Global vs Min-F2.

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

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# 0. CONFIG & PATHS
# ═════════════════════════════════════════════════════════════════════════════
REPO_ROOT    = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIG_DIR      = RESULTS_ROOT / "figures" / "main_experiments"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["TextCNN", "BERT", "DistilBERT", "ModernBERT", "SecBERT", "SecureBERT", "CySecBERT"]
DATASETS = ["cti-mitre", "joint", "tram"]
SCENARIOS = ["A0", "G0", "B1", "G1", "B2"]

DATASET_TITLES = {
    "cti-mitre": "CTI-to-MITRE (188 labels)",
    "joint":     "Joint CTI+TRAM (188 labels)",
    "tram":      "TRAM (50 labels)",
}

DATASET_SHORT = {
    "cti-mitre": "CTI-MITRE",
    "joint":     "Joint",
    "tram":      "TRAM",
}

SCENARIO_TITLES = {
    "A0": "Baseline",
    "G0": "GenEDA (1-Stage)",
    "B1": "CyberEDA (1-Stage)",
    "G1": "GenEDA (2-Stage)",
    "B2": "CyberEDA (2-Stage)",
}

MODEL_COLORS = {
    "TextCNN":    "#64748B",  # Slate
    "BERT":       "#D97706",  # Amber
    "DistilBERT": "#0284C7",  # Sky Blue
    "ModernBERT": "#059669",  # Emerald Green
    "SecBERT":    "#EA580C",  # Orange-Red
    "SecureBERT": "#2563EB",  # Royal Blue
    "CySecBERT":  "#7C3AED",  # Purple
}

SCENARIO_COLORS = {
    "A0": "#64748B",  # Slate Gray
    "G0": "#F59E0B",  # Warm Amber
    "B1": "#10B981",  # Emerald Green
    "G1": "#D97706",  # Dark Ochre
    "B2": "#047857",  # Deep Teal
}

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER FOR ALL 105 RUNS
# ═════════════════════════════════════════════════════════════════════════════
def load_all_benchmark_data():
    """Scans and parses all 105 experimental runs."""
    model_dirs = {
        "BERT":       RESULTS_ROOT / "BErt",
        "DistilBERT": RESULTS_ROOT / "DistilBERT",
        "CySecBERT":  RESULTS_ROOT / "CySecBERT results",
        "ModernBERT": RESULTS_ROOT / "MordernBert",
        "SecBERT":    RESULTS_ROOT / "SecBERT",
        "SecureBERT": RESULTS_ROOT / "secureBert",
        "TextCNN":    RESULTS_ROOT / "TextCNN",
    }

    ds_aliases = {
        "cti-mitre": ["cti-mitre", "cti_to_mitre", "cti-to-mitre", "cti_mitre", "cti", "mitre", "ctitomitre"],
        "joint":     ["joint"],
        "tram":      ["tram"],
    }

    records = []

    for m in MODELS:
        mdir = model_dirs[m]
        if not mdir.exists():
            continue
        for d in DATASETS:
            aliases = ds_aliases[d]
            for sc in SCENARIOS:
                sc_aliases = [sc]
                if sc == "B2":
                    sc_aliases.extend(["B2_E1", "B2-E1", "B2_e1", "b2_e1"])

                matched_folder = None
                for mf in mdir.rglob("metrics.json"):
                    p = mf.parent
                    full_str = str(p.relative_to(mdir)).lower()
                    sc_match = any(sc_a.lower() in full_str for sc_a in sc_aliases)
                    ds_match = any(a in full_str for a in aliases)
                    if sc_match and ds_match:
                        matched_folder = p
                        break

                if matched_folder and (matched_folder / "metrics.json").exists():
                    try:
                        m_data = json.loads((matched_folder / "metrics.json").read_text(encoding="utf-8"))
                        tg = m_data.get("test_global", {})
                        tm = m_data.get("test_minority_f2", {})
                        if tm is None:
                            tm = {}

                        tier_stats_g = {"Tail": 0.0, "Medium": 0.0, "Head": 0.0}
                        tier_stats_m = {"Tail": 0.0, "Medium": 0.0, "Head": 0.0}

                        pl_g_path = matched_folder / "per_label_global.csv"
                        pl_m_path = matched_folder / "per_label_minority_f2.csv"
                        pl_alt_path = matched_folder / "per_label_metrics.csv"
                        fg_path = matched_folder / "frequency_group_performance.csv"

                        # Parse Global Tier Stats
                        if pl_g_path.exists():
                            try:
                                df_plg = pd.read_csv(pl_g_path)
                                for b in ["Tail", "Medium", "Head"]:
                                    sub_b = df_plg[df_plg["Support_Bucket"] == b]
                                    if len(sub_b) > 0:
                                        tier_stats_g[b] = float(sub_b["F1"].mean())
                            except Exception:
                                pass
                        elif pl_alt_path.exists():
                            try:
                                df_pla = pd.read_csv(pl_alt_path)
                                group_col = "Frequency_Group" if "Frequency_Group" in df_pla.columns else "Support_Bucket"
                                for b in ["Tail", "Medium", "Head"]:
                                    sub_b = df_pla[df_pla[group_col] == b]
                                    if len(sub_b) > 0:
                                        tier_stats_g[b] = float(sub_b["F1"].mean())
                            except Exception:
                                pass
                        elif fg_path.exists():
                            try:
                                df_fg = pd.read_csv(fg_path)
                                for _, row in df_fg.iterrows():
                                    grp = row["Frequency_Group"]
                                    if grp in tier_stats_g:
                                        tier_stats_g[grp] = float(row["mean_f1"])
                            except Exception:
                                pass

                        # Parse Minority-F2 Tier Stats (fallback to Global for baseline / models without minf2)
                        if pl_m_path.exists():
                            try:
                                df_plm = pd.read_csv(pl_m_path)
                                for b in ["Tail", "Medium", "Head"]:
                                    sub_b = df_plm[df_plm["Support_Bucket"] == b]
                                    if len(sub_b) > 0:
                                        tier_stats_m[b] = float(sub_b["F1"].mean())
                            except Exception:
                                pass
                        else:
                            tier_stats_m = tier_stats_g.copy()

                        # Parse Zero-F1 Count
                        z_cnt_g = tg.get("zero_f1_count", None)
                        if z_cnt_g is None:
                            if pl_alt_path.exists():
                                try:
                                    df_pla = pd.read_csv(pl_alt_path)
                                    z_cnt_g = int((df_pla["F1"] == 0).sum())
                                except Exception:
                                    z_cnt_g = 0
                            elif pl_g_path.exists():
                                try:
                                    df_plg = pd.read_csv(pl_g_path)
                                    z_cnt_g = int((df_plg["F1"] == 0).sum())
                                except Exception:
                                    z_cnt_g = 0
                            else:
                                z_cnt_g = 0

                        z_cnt_m = tm.get("zero_f1_count", z_cnt_g)

                        rec = {
                            "model":                       m,
                            "dataset":                     d,
                            "scenario":                    sc,
                            "folder":                      str(matched_folder),
                            "tau_global":                  m_data.get("selected_global_threshold", None),
                            "tau_minority":                m_data.get("selected_minority_threshold", None),
                            # Global Metrics
                            "macro_f1_global":             tg.get("macro_f1", 0.0),
                            "macro_recall_global":         tg.get("macro_recall", 0.0),
                            "macro_precision_global":      tg.get("macro_precision", 0.0),
                            "micro_f1_global":             tg.get("micro_f1", 0.0),
                            "zero_f1_global":              z_cnt_g,
                            "hit3_global":                 tg.get("hit_at_3", 0.0),
                            "hit5_global":                 tg.get("hit_at_5", 0.0),
                            "mrr_global":                  tg.get("mrr", 0.0),
                            "map_global":                  tg.get("map", 0.0),
                            # Minority-F2 Metrics
                            "has_minority_f2":             bool(tm),
                            "macro_f1_minf2":              tm.get("macro_f1", tg.get("macro_f1", 0.0)),
                            "macro_recall_minf2":          tm.get("macro_recall", tg.get("macro_recall", 0.0)),
                            "macro_precision_minf2":       tm.get("macro_precision", tg.get("macro_precision", 0.0)),
                            "micro_f1_minf2":              tm.get("micro_f1", tg.get("micro_f1", 0.0)),
                            "zero_f1_minf2":               z_cnt_m,
                            "hit3_minf2":                  tm.get("hit_at_3", tg.get("hit_at_3", 0.0)),
                            "hit5_minf2":                  tm.get("hit_at_5", tg.get("hit_at_5", 0.0)),
                            "mrr_minf2":                   tm.get("mrr", tg.get("mrr", 0.0)),
                            "map_minf2":                   tm.get("map", tg.get("map", 0.0)),
                            # Computation
                            "training_seconds":            m_data.get("training_seconds", 0.0),
                            "peak_vram_mb":                m_data.get("peak_vram_mb", 0.0),
                            # Tier breakdown
                            "tier_tail_f1_global":         tier_stats_g["Tail"],
                            "tier_med_f1_global":          tier_stats_g["Medium"],
                            "tier_head_f1_global":         tier_stats_g["Head"],
                            "tier_tail_f1_minf2":          tier_stats_m["Tail"],
                            "tier_med_f1_minf2":           tier_stats_m["Medium"],
                            "tier_head_f1_minf2":          tier_stats_m["Head"],
                        }
                        records.append(rec)
                    except Exception as e:
                        print(f"[ERR] Reading {matched_folder}: {e}")

    df = pd.DataFrame(records)
    print(f"[DATA LOADER] Successfully loaded {len(df)}/105 runs.")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. ACADEMIC TABLE IMAGE RENDERER
# ═════════════════════════════════════════════════════════════════════════════
def render_academic_table_image(headers, cell_text, col_widths, title, footnote,
                                output_path, figsize=(15.5, 9.0),
                                highlight_cols=None):
    """Renders a publication-grade table image with strict academic styling."""
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
    if num_rows <= 15:
        font_size = 8.8
        row_scale = 1.7
    elif num_rows <= 25:
        font_size = 8.0
        row_scale = 1.45
    else:
        font_size = 6.8
        row_scale = 1.20
    table.set_fontsize(font_size)
    table.scale(1.0, row_scale)

    # Style Header Row
    for col_idx in range(len(headers)):
        c = table[(0, col_idx)]
        c.set_facecolor("#1E3A8A")  # Deep Navy
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

    # Table Title (Academic Convention: Above Table)
    plt.title(title, fontsize=12.5, fontweight="bold", color="#0F172A", pad=25, y=0.98)

    # Academic Footnote (Below Table)
    plt.figtext(0.5, 0.02, footnote, ha="center", fontsize=8.0, color="#475569", style="italic", linespacing=1.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Table Image saved: {output_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. GENERATE 7 PUBLICATION TABLE IMAGES
# ═════════════════════════════════════════════════════════════════════════════

def generate_table01_main_benchmark_dataset(df, dataset_name, output_path):
    df_sub = df[df["dataset"] == dataset_name].copy()
    headers = [
        "Model", "Scenario",
        "Threshold\n[Glob / MinF2]",
        "Macro-F1 (%)\n[Global]", "Macro-F1 (%)\n[Minority-F2]",
        "Macro-Rec (%)\n[Global]", "Macro-Rec (%)\n[Minority-F2]",
        "Zero-F1\n[Global]", "Zero-F1\n[Minority-F2]",
        "Micro-F1 (%)\n[Minority-F2]"
    ]
    col_widths = [0.12, 0.15, 0.11, 0.10, 0.11, 0.10, 0.11, 0.09, 0.10, 0.11]
    cell_text = []

    for m in MODELS:
        for sc in SCENARIOS:
            row = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == sc)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            tg = f"{r['tau_global']:.2f}" if pd.notna(r["tau_global"]) else "--"
            tm = f"{r['tau_minority']:.2f}" if pd.notna(r["tau_minority"]) else "--"
            thresh_str = f"{tg} / {tm}"

            fg = f"{r['macro_f1_global']*100:.2f}%"
            fm = f"{r['macro_f1_minf2']*100:.2f}%" + ("*" if not r["has_minority_f2"] else "")
            rg = f"{r['macro_recall_global']*100:.2f}%"
            rm = f"{r['macro_recall_minf2']*100:.2f}%" + ("*" if not r["has_minority_f2"] else "")
            zg = str(int(r["zero_f1_global"]))
            zm = str(int(r["zero_f1_minf2"])) + ("*" if not r["has_minority_f2"] else "")
            mi = f"{r['micro_f1_minf2']*100:.2f}%" + ("*" if not r["has_minority_f2"] else "")

            cell_text.append([m, SCENARIO_TITLES[sc], thresh_str, fg, fm, rg, rm, zg, zm, mi])

    title = f"TABLE I: Comprehensive Benchmark on {DATASET_TITLES[dataset_name]} (Global Threshold vs. Minority-F2)"
    footnote = (
        f"* Notes: Evaluated on {DATASET_SHORT[dataset_name]} test split. Baseline scenario (A0) evaluated under standard Global Threshold (τ_g); Augmentation scenarios (G0, B1, G1, B2) evaluated under Minority-F2 Threshold (τ_m).\n"
        "* Global τ_g tuned on Val Macro-F1; Minority τ_m tuned on Val Minority-F2. Highlighted columns indicate Minority-F2 optimization."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(16.8, 12.8), highlight_cols=[4, 6, 8]
    )


def generate_table02_thresholding_comparison(df, output_path):
    """Table 2: Direct Thresholding Comparison for both GenericEDA (G0) and CyberEDA (B1)."""
    df_sub = df[df["dataset"] == "joint"].copy()
    headers = [
        "Architecture", "Scenario",
        "Global Thresh\n(τ_g)", "Min-F2 Thresh\n(τ_m)",
        "Recall (%)\n[Global]", "Recall (%)\n[Min-F2]", "Δ Recall\n(Gain pp)",
        "Zero-F1\n[Global]", "Zero-F1\n[Min-F2]", "Δ Zero-F1\n(Rescued)",
        "Macro-F1 (%)\n[Global]", "Macro-F1 (%)\n[Min-F2]"
    ]
    col_widths = [0.12, 0.13, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.08, 0.08, 0.08]
    cell_text = []

    for m in MODELS:
        for sc in ["G0", "B1"]:
            row = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == sc)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            if not r["has_minority_f2"]:
                continue
            tg = r["tau_global"]; tm = r["tau_minority"]
            rg = r["macro_recall_global"] * 100; rm = r["macro_recall_minf2"] * 100
            d_rec = rm - rg
            zg = int(r["zero_f1_global"]); zm = int(r["zero_f1_minf2"])
            d_zg = zm - zg
            fg = r["macro_f1_global"] * 100; fm = r["macro_f1_minf2"] * 100

            cell_text.append([
                m, SCENARIO_TITLES[sc],
                f"{tg:.2f}" if pd.notna(tg) else "--",
                f"{tm:.2f}" if pd.notna(tm) else "--",
                f"{rg:.2f}%", f"{rm:.2f}%", f"+{d_rec:.2f} pp",
                str(zg), str(zm), f"{d_zg:+d} labels",
                f"{fg:.2f}%", f"{fm:.2f}%"
            ])

    title = "TABLE II: Cross-Model Efficacy of Minority-F2 Optimization vs. Global Thresholding (Joint Dataset, GenericEDA vs. CyberEDA)"
    footnote = (
        "* Notes: Evaluated on Joint test split (188 labels). Compares Global Threshold (τ_g) vs. Minority-F2 Threshold (τ_m).\n"
        "* Calculation: ΔRecall = Recall_MinF2 - Recall_Global; ΔZero-F1 = Zero-F1_MinF2 - Zero-F1_Global (negative denotes rescued techniques)."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(16.5, 7.5), highlight_cols=[6, 9]
    )


def generate_table03_training_overhead(df, output_path):
    """Table 3: Computational Cost, Training Overhead, and Augmentation Saturation (Joint Dataset)."""
    df_sub = df[df["dataset"] == "joint"].copy()
    headers = [
        "Architecture",
        "Base A0\nTime (s)", "GenEDA G0\nTime (s)", "CyberEDA B1\nTime (s)",
        "Overhead G0\n(%)", "Overhead B1\n(%)",
        "Macro-F1\nBase A0 (%)", "Macro-F1\nG0 (%)", "Macro-F1\nB1 (%)",
        "Efficiency G0\n(ΔF1/Δt x1e3)", "Efficiency B1\n(ΔF1/Δt x1e3)", "Peak VRAM\n(MB)"
    ]
    col_widths = [0.11, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.08]
    cell_text = []

    for m in MODELS:
        r_a0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")]
        r_g0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")]
        r_b1 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")]
        if len(r_a0) == 0 or len(r_g0) == 0 or len(r_b1) == 0:
            continue
        a0 = r_a0.iloc[0]; g0 = r_g0.iloc[0]; b1 = r_b1.iloc[0]
        t0 = a0["training_seconds"]; tg = g0["training_seconds"]; tb = b1["training_seconds"]
        f0 = a0["macro_f1_global"] * 100; fg = g0["macro_f1_minf2"] * 100; fb = b1["macro_f1_minf2"] * 100

        ov_g = ((tg - t0) / t0 * 100) if t0 > 0 else 0.0
        ov_b = ((tb - t0) / t0 * 100) if t0 > 0 else 0.0
        eff_g = ((fg - f0) / (tg - t0) * 1000) if (tg - t0) > 0 else 0.0
        eff_b = ((fb - f0) / (tb - t0) * 1000) if (tb - t0) > 0 else 0.0
        vram = b1["peak_vram_mb"]

        cell_text.append([
            m, f"{t0:.1f}s", f"{tg:.1f}s", f"{tb:.1f}s",
            f"+{ov_g:.1f}%", f"+{ov_b:.1f}%",
            f"{f0:.2f}%", f"{fg:.2f}%", f"{fb:.2f}%",
            f"{eff_g:.2f}", f"{eff_b:.2f}", f"{vram:.0f} MB"
        ])

    title = "TABLE III: Computational Cost, Training Overhead, and Augmentation Efficiency (Joint Dataset)"
    footnote = (
        "* Notes: Evaluated on Joint dataset. A0 = Baseline, G0 = GenericEDA (1-Stage), B1 = CyberEDA (1-Stage). Hardware: Single NVIDIA GPU, identical batch size.\n"
        "* Calculation: Training Overhead (%) = [(T_EDA - T_Base) / T_Base] × 100%; Efficiency = [(Macro-F1_EDA - Macro-F1_Base) / (T_EDA - T_Base)] × 1,000."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(16.5, 5.2), highlight_cols=[4, 5, 9, 10]
    )


def generate_table04_frequency_tier(df, output_path):
    df_sub = df[df["dataset"] == "joint"].copy()
    headers = [
        "Architecture", "Scenario",
        "Tail F1 (%)\n[Global]", "Tail F1 (%)\n[Min-F2]",
        "Medium F1 (%)\n[Global]", "Medium F1 (%)\n[Min-F2]",
        "Head F1 (%)\n[Global]", "Head F1 (%)\n[Min-F2]"
    ]
    col_widths = [0.15, 0.17, 0.11, 0.11, 0.11, 0.11, 0.12, 0.12]
    cell_text = []

    for m in MODELS:
        for sc in ["A0", "G0", "B1"]:
            row = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == sc)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            tg_tail = f"{r['tier_tail_f1_global']*100:.2f}%"
            tm_tail = f"{r['tier_tail_f1_minf2']*100:.2f}%"
            tg_med  = f"{r['tier_med_f1_global']*100:.2f}%"
            tm_med  = f"{r['tier_med_f1_minf2']*100:.2f}%"
            tg_head = f"{r['tier_head_f1_global']*100:.2f}%"
            tm_head = f"{r['tier_head_f1_minf2']*100:.2f}%"
            cell_text.append([m, SCENARIO_TITLES[sc], tg_tail, tm_tail, tg_med, tm_med, tg_head, tm_head])

    title = "TABLE IV: ATT&CK Technique Classification Breakdown Across Sample Frequency Tiers (Joint Dataset)"
    footnote = (
        "* Notes: Evaluated on Joint test split (188 labels). Baseline scenario (A0) evaluated under standard Global Threshold; Augmentation scenarios (G0, B1) evaluated under Minority-F2 Threshold.\n"
        "* Calculation: Tier Mean F1 = Mean F1 score across classes within designated frequency tier (Tail: <30, Medium: 30-100, Head: >100 samples)."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(16.0, 10.5), highlight_cols=[3, 5, 7]
    )


def generate_table05_retrieval_ranking(df, output_path):
    df_sub = df[df["dataset"] == "joint"].copy()
    headers = [
        "Architecture", "Scenario",
        "Hit@3 (%)", "Hit@5 (%)", "MRR (Mean Reciprocal Rank)", "MAP (Mean Average Precision)"
    ]
    col_widths = [0.18, 0.16, 0.16, 0.16, 0.17, 0.17]
    cell_text = []

    for m in MODELS:
        for sc in ["A0", "G0", "B1"]:
            row = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == sc)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            h3 = f"{r['hit3_minf2']*100:.2f}%"
            h5 = f"{r['hit5_minf2']*100:.2f}%"
            mrr = f"{r['mrr_minf2']:.4f}"
            mapp = f"{r['map_minf2']:.4f}"
            cell_text.append([m, SCENARIO_TITLES[sc], h3, h5, mrr, mapp])

    title = "TABLE V: Operational Recommendation Quality for Security Operations Center (SOC) Incident Response"
    footnote = (
        "* Notes: Evaluated on Joint test split (4,001 CTI reports) across Baseline (A0), GenericEDA (G0), and CyberEDA (B1).\n"
        "* Baseline scenario (A0) evaluated under standard Global Threshold; Augmentation scenarios (G0, B1) evaluate under Minority-F2 Threshold."
    )
    render_academic_table_image(
        headers, cell_text, col_widths, title, footnote, output_path,
        figsize=(14.5, 9.5), highlight_cols=[2, 3]
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. FIGURES (FIGURE 1 to 9 - Grouped Bars Only, Prominent Titles & Bottom Notes)
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig01_macro_f1_benchmark_minf2(df, output_path):
    """Core Figure 1: Comprehensive Scenario Benchmark across 7 models x 5 scenarios under Minority-F2 Threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), dpi=300, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    x = np.arange(len(MODELS))
    w = 0.16

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        df_ds = df[df["dataset"] == ds]

        for s_idx, sc in enumerate(SCENARIOS):
            vals = []
            for m in MODELS:
                r = df_ds[(df_ds["model"] == m) & (df_ds["scenario"] == sc)]
                vals.append(r.iloc[0]["macro_f1_minf2"] * 100 if len(r) > 0 else 0.0)
            offset = (s_idx - 2) * w
            ax.bar(x + offset, vals, w, label=SCENARIO_TITLES[sc], color=SCENARIO_COLORS[sc], alpha=0.92)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=8.5, rotation=25)
        ax.set_title(f"({chr(97+i)}) {DATASET_TITLES[ds]}", fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Macro-F1 Score (%)", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 92)

    axes[0].legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.5, loc="upper left")

    plt.suptitle("Figure 1: Comprehensive Scenario Benchmark Across 7 Deep Learning Architectures",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.12)

    caption = (
        "Notes: Augmentation scenarios (B1, B2, G0, G1) evaluated under Minority-F2 Threshold (τ_m); Baseline scenario (A0) evaluated under standard Global Threshold (τ_g)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig02_tier_breakdown_comparison(df, output_path):
    """Core Figure 2: ATT&CK Technique Frequency Tier Breakdown: Baseline vs GenericEDA vs CyberEDA."""
    df_sub = df[df["dataset"] == "joint"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), dpi=300, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    tiers = [
        ("(a) Tail Tier (<30 samples: 99 labels)", "tier_tail_f1_minf2"),
        ("(b) Medium Tier (30-100: 51 labels)", "tier_med_f1_minf2"),
        ("(c) Head Tier (>100: 38 labels)", "tier_head_f1_minf2")
    ]

    x = np.arange(len(MODELS))
    w = 0.26

    for i, (t_title, t_col) in enumerate(tiers):
        ax = axes[i]
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        vals_a0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0][t_col.replace("_minf2", "_global")] * 100 for m in MODELS]
        vals_g0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0][t_col] * 100 for m in MODELS]
        vals_b1 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0][t_col] * 100 for m in MODELS]

        ax.bar(x - w, vals_a0, w, label="Baseline (A0)", color="#64748B", alpha=0.9)
        ax.bar(x, vals_g0, w, label="GenericEDA (G0)", color="#D97706", alpha=0.9)
        ax.bar(x + w, vals_b1, w, label="CyberEDA (B1)", color="#10B981", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=8.5, rotation=20)
        ax.set_title(t_title, fontsize=10.5, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Mean F1 Score (%)", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 95)

    axes[0].legend(frameon=True, fontsize=8.5, loc="upper left")

    plt.suptitle("Figure 2: ATT&CK Technique Frequency Tier Breakdown on Joint Dataset: Baseline vs. GenericEDA vs. CyberEDA",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.12)

    caption = (
        "Notes: Augmentation scenarios (G0, B1) evaluated under Minority-F2 Threshold; Baseline scenario (A0) evaluated under standard Global Threshold.\n"
        "Calculation: Tier Mean F1 = Mean F1 score across classes within designated frequency tier (Tail: <30, Medium: 30-100, Head: >100 samples)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig03_macro_micro_gap(df, output_path):
    """Core Figure 3: Macro vs Micro Imbalance Gap for Baseline A0 vs GenericEDA G0 vs CyberEDA B1."""
    df_sub = df[df["dataset"] == "joint"].copy()
    fig, ax = plt.subplots(figsize=(13.5, 5.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    gap_a0 = []; gap_g0 = []; gap_b1 = []; models_valid = []

    for m in MODELS:
        r_a0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")]
        r_g0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")]
        r_b1 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")]
        if len(r_a0) > 0 and len(r_g0) > 0 and len(r_b1) > 0:
            g_a0 = (r_a0.iloc[0]["micro_f1_global"] - r_a0.iloc[0]["macro_f1_global"]) * 100
            g_g0 = (r_g0.iloc[0]["micro_f1_minf2"] - r_g0.iloc[0]["macro_f1_minf2"]) * 100
            g_b1 = (r_b1.iloc[0]["micro_f1_minf2"] - r_b1.iloc[0]["macro_f1_minf2"]) * 100
            gap_a0.append(g_a0)
            gap_g0.append(g_g0)
            gap_b1.append(g_b1)
            models_valid.append(m)

    x = np.arange(len(models_valid))
    w = 0.26

    r1 = ax.bar(x - w, gap_a0, w, label="Baseline (A0) Gap", color="#EF4444", alpha=0.90)
    r2 = ax.bar(x, gap_g0, w, label="GenericEDA (G0) Gap", color="#F59E0B", alpha=0.90)
    r3 = ax.bar(x + w, gap_b1, w, label="CyberEDA (B1) Gap", color="#10B981", alpha=0.90)

    for r in r1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.6, f"{r.get_height():.1f}%", ha="center", fontsize=7.5)
    for r in r2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.6, f"{r.get_height():.1f}%", ha="center", fontsize=7.5)
    for r in r3:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.6, f"{r.get_height():.1f}%", ha="center", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models_valid, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Micro-F1 minus Macro-F1 Gap (pp)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=True, fontsize=9.0)
    ax.set_ylim(0, max(gap_a0) * 1.20)

    plt.suptitle("Figure 3: Class Imbalance Evidence: Contraction of Macro- vs. Micro-F1 Disparity on Joint Dataset",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.08, right=0.96)

    caption = (
        "Notes: Augmentation scenarios (G0, B1) evaluated under Minority-F2 Threshold (τ_m); Baseline scenario (A0) evaluated under standard Global Threshold (τ_g).\n"
        "Calculation: Imbalance Gap = Micro-F1 - Macro-F1 (percentage points)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig04_cyber_vs_generic_head_to_head(df, output_path):
    """Figure 4: Neutral Head-to-Head Comparison: 1-Stage (B1 vs G0) & 2-Stage (B2 vs G1)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 5.8), dpi=300, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    x = np.arange(len(MODELS))
    w = 0.24

    colors_ds = {
        "cti-mitre": "#2563EB",  # Royal Blue
        "joint":     "#059669",  # Emerald Green
        "tram":      "#D97706",  # Amber
    }

    all_deltas = []

    # Panel (a): 1-Stage Comparison (B1 minus G0)
    ax1.set_facecolor("#FFFFFF")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.axhline(0, color="#0F172A", linewidth=1.0, linestyle="-")

    for i, ds in enumerate(DATASETS):
        df_ds = df[df["dataset"] == ds]
        deltas = []
        for m in MODELS:
            r_b1 = df_ds[(df_ds["model"] == m) & (df_ds["scenario"] == "B1")]
            r_g0 = df_ds[(df_ds["model"] == m) & (df_ds["scenario"] == "G0")]
            if len(r_b1) > 0 and len(r_g0) > 0:
                f_b1 = r_b1.iloc[0]["macro_f1_minf2"] * 100
                f_g0 = r_g0.iloc[0]["macro_f1_minf2"] * 100
                deltas.append(f_b1 - f_g0)
            else:
                deltas.append(0.0)

        all_deltas.extend(deltas)
        offset = (i - 1) * w
        rects = ax1.bar(x + offset, deltas, w, label=f"{DATASET_SHORT[ds]}",
                        color=colors_ds[ds], alpha=0.92)

        for rect in rects:
            val = rect.get_height()
            va_align = "bottom" if val >= 0 else "top"
            offset_y = 0.20 if val >= 0 else -0.45
            ax1.text(rect.get_x() + rect.get_width()/2, val + offset_y,
                     f"{val:+.1f}", ha="center", va=va_align, fontsize=7.0, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(MODELS, fontsize=8.5, rotation=15)
    ax1.set_ylabel("Macro-F1 Margin (pp)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) 1-Stage: CyberEDA (B1) minus GenericEDA (G0)", fontsize=10.5, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.2, loc="upper right")

    # Panel (b): 2-Stage Comparison (B2 minus G1)
    ax2.set_facecolor("#FFFFFF")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.axhline(0, color="#0F172A", linewidth=1.0, linestyle="-")

    for i, ds in enumerate(DATASETS):
        df_ds = df[df["dataset"] == ds]
        deltas2 = []
        for m in MODELS:
            r_b2 = df_ds[(df_ds["model"] == m) & (df_ds["scenario"] == "B2")]
            r_g1 = df_ds[(df_ds["model"] == m) & (df_ds["scenario"] == "G1")]
            if len(r_b2) > 0 and len(r_g1) > 0:
                f_b2 = r_b2.iloc[0]["macro_f1_minf2"] * 100
                f_g1 = r_g1.iloc[0]["macro_f1_minf2"] * 100
                deltas2.append(f_b2 - f_g1)
            else:
                deltas2.append(0.0)

        all_deltas.extend(deltas2)
        offset = (i - 1) * w
        rects2 = ax2.bar(x + offset, deltas2, w, label=f"{DATASET_SHORT[ds]}",
                         color=colors_ds[ds], alpha=0.92)

        for rect in rects2:
            val = rect.get_height()
            va_align = "bottom" if val >= 0 else "top"
            offset_y = 0.20 if val >= 0 else -0.45
            ax2.text(rect.get_x() + rect.get_width()/2, val + offset_y,
                     f"{val:+.1f}", ha="center", va=va_align, fontsize=7.0, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(MODELS, fontsize=8.5, rotation=15)
    ax2.set_title("(b) 2-Stage: CyberEDA (B2) minus GenericEDA (G1)", fontsize=10.5, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.2, loc="upper right")

    min_val = min(all_deltas) - 1.2
    max_val = max(all_deltas) + 1.2
    ax1.set_ylim(min_val, max_val)

    plt.suptitle("Figure 4: Neutral Head-to-Head Performance Margin: CyberEDA vs. GenericEDA",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.10)

    caption = (
        "Notes: Evaluated under Minority-F2 Threshold (τ_m).\n"
        "Calculation: Margin = Macro-F1(CyberEDA) - Macro-F1(GenericEDA) (percentage points; positive indicates CyberEDA lead, negative indicates GenericEDA lead)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig05_zf1_reduction(df, output_path):
    """Figure 5: Number of Missed Attack Techniques (Zero-F1 Count): Baseline vs GenericEDA vs CyberEDA."""
    df_sub = df[df["dataset"] == "joint"].copy()

    zf1_a0_g = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]["zero_f1_global"] for m in MODELS]
    zf1_g0_g = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]["zero_f1_global"] for m in MODELS]
    zf1_b1_g = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]["zero_f1_global"] for m in MODELS]

    zf1_a0_m = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]["zero_f1_minf2"] for m in MODELS]
    zf1_g0_m = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]["zero_f1_minf2"] for m in MODELS]
    zf1_b1_m = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]["zero_f1_minf2"] for m in MODELS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 5.8), dpi=300, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    x = np.arange(len(MODELS))
    w = 0.26

    # Panel (a): Under Global Threshold
    r1_a = ax1.bar(x - w, zf1_a0_g, w, label="Baseline (A0)", color="#64748B", alpha=0.92)
    r2_a = ax1.bar(x, zf1_g0_g, w, label="GenericEDA (G0)", color="#F59E0B", alpha=0.92)
    r3_a = ax1.bar(x + w, zf1_b1_g, w, label="CyberEDA (B1)", color="#10B981", alpha=0.92)

    for rect in list(r1_a) + list(r2_a) + list(r3_a):
        val = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, val + 1.2, f"{int(round(val))}",
                 ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1E293B")

    ax1.set_xticks(x); ax1.set_xticklabels(MODELS, fontsize=8.5, rotation=15)
    ax1.set_ylabel("Missed Attack Techniques (Zero-F1 Count)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Missed Techniques under Global Threshold (τ_g)", fontsize=10.5, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5)

    # Panel (b): Under Minority-F2 Threshold
    r1_b = ax2.bar(x - w, zf1_a0_m, w, label="Baseline (A0)", color="#64748B", alpha=0.92)
    r2_b = ax2.bar(x, zf1_g0_m, w, label="GenericEDA (G0)", color="#F59E0B", alpha=0.92)
    r3_b = ax2.bar(x + w, zf1_b1_m, w, label="CyberEDA (B1)", color="#10B981", alpha=0.92)

    for rect in list(r1_b) + list(r2_b) + list(r3_b):
        val = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2, val + 1.2, f"{int(round(val))}",
                 ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1E293B")

    ax2.set_xticks(x); ax2.set_xticklabels(MODELS, fontsize=8.5, rotation=15)
    ax2.set_title("(b) Missed Techniques under Minority-F2 Threshold (τ_m)", fontsize=10.5, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.5)
    ax1.set_ylim(0, max(zf1_a0_g) * 1.18)
    ax2.set_ylim(0, max(zf1_a0_g) * 1.18)

    plt.suptitle("Figure 5: Number of Missed Attack Techniques (Zero-F1 Count) on Joint Dataset: Baseline vs. GenericEDA vs. CyberEDA",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.10)

    caption = (
        "Notes: Panel (a) evaluated under Global Threshold (τ_g); Panel (b) evaluated under Minority-F2 Threshold (τ_m).\n"
        "Calculation: Zero-F1 Count = Total number of ATT&CK technique classes with test F1 = 0.0 (complete false negatives)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig06_recall_and_precision_tradeoff(df, output_path):
    """Figure 6: Macro-Recall Gain & Precision/F1 Trade-off: Baseline vs GenericEDA vs CyberEDA."""
    df_sub = df[df["dataset"] == "joint"].copy()

    rec_a0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]["macro_recall_global"] * 100 for m in MODELS]
    rec_g0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]["macro_recall_minf2"] * 100 for m in MODELS]
    rec_b1 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]["macro_recall_minf2"] * 100 for m in MODELS]

    prec_a0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]["macro_precision_global"] * 100 for m in MODELS]
    prec_g0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]["macro_precision_minf2"] * 100 for m in MODELS]
    prec_b1 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]["macro_precision_minf2"] * 100 for m in MODELS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 5.8), dpi=300, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    x = np.arange(len(MODELS))
    w = 0.26

    # Panel (a): Macro-Recall
    ax1.bar(x - w, rec_a0, w, label="Baseline (A0)", color="#64748B", alpha=0.92)
    ax1.bar(x, rec_g0, w, label="GenericEDA (G0)", color="#F59E0B", alpha=0.92)
    ax1.bar(x + w, rec_b1, w, label="CyberEDA (B1)", color="#10B981", alpha=0.92)
    ax1.set_xticks(x); ax1.set_xticklabels(MODELS, fontsize=8.5, rotation=15)
    ax1.set_ylabel("Metric Score (%)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Macro-Recall (Threat Indicator Detection Sensitivity)", fontsize=10.5, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5)

    # Panel (b): Macro-Precision
    ax2.bar(x - w, prec_a0, w, label="Baseline (A0)", color="#64748B", alpha=0.92)
    ax2.bar(x, prec_g0, w, label="GenericEDA (G0)", color="#F59E0B", alpha=0.92)
    ax2.bar(x + w, prec_b1, w, label="CyberEDA (B1)", color="#10B981", alpha=0.92)
    ax2.set_xticks(x); ax2.set_xticklabels(MODELS, fontsize=8.5, rotation=15)
    ax2.set_title("(b) Macro-Precision (False Alarm Suppression)", fontsize=10.5, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.5)
    ax1.set_ylim(0, 95)

    plt.suptitle("Figure 6: Macro-Recall Gain and Macro-Precision Trade-off on Joint Dataset: Baseline vs. GenericEDA vs. CyberEDA",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.10)

    caption = (
        "Notes: Augmentation scenarios (G0, B1) evaluated under Minority-F2 Threshold (τ_m); Baseline scenario (A0) evaluated under standard Global Threshold (τ_g).\n"
        "Compares threat detection sensitivity (Macro-Recall) against false alarm suppression (Macro-Precision)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig07_macro_f1_benchmark_global(df, output_path):
    """Figure 7: Reference Macro-F1 across 7 models x 5 scenarios under Global Threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), dpi=300, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    x = np.arange(len(MODELS))
    w = 0.16

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        df_ds = df[df["dataset"] == ds]

        for s_idx, sc in enumerate(SCENARIOS):
            vals = []
            for m in MODELS:
                r = df_ds[(df_ds["model"] == m) & (df_ds["scenario"] == sc)]
                vals.append(r.iloc[0]["macro_f1_global"] * 100 if len(r) > 0 else 0.0)
            offset = (s_idx - 2) * w
            ax.bar(x + offset, vals, w, label=SCENARIO_TITLES[sc], color=SCENARIO_COLORS[sc], alpha=0.92)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=8.5, rotation=25)
        ax.set_title(f"({chr(97+i)}) {DATASET_TITLES[ds]}", fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Macro-F1 Score (%)", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 92)

    axes[0].legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.5, loc="upper left")

    plt.suptitle("Figure 7: Reference Macro-F1 Performance Across 7 Architectures under Standard Global Threshold",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.12)

    caption = (
        "Notes: Evaluated under uniform Global Threshold (τ_g tuned on validation Macro-F1 without minority class re-weighting)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig08_training_time_and_overhead(df, output_path):
    """Figure 8: Two-panel bar chart for Training Time & Augmentation Overhead (A0, G0, B1, G1, B2)."""
    df_sub = df[df["dataset"] == "joint"].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 5.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel (a): Training Duration: Baseline A0 vs GenericEDA G0 vs CyberEDA B1
    x = np.arange(len(MODELS))
    w = 0.26
    times_a0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]["training_seconds"] for m in MODELS]
    times_g0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]["training_seconds"] for m in MODELS]
    times_b1 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]["training_seconds"] for m in MODELS]

    ax1.bar(x - w, times_a0, w, label="Baseline (A0)", color="#64748B", alpha=0.92)
    ax1.bar(x, times_g0, w, label="GenericEDA (G0)", color="#F59E0B", alpha=0.92)
    ax1.bar(x + w, times_b1, w, label="CyberEDA (B1)", color="#10B981", alpha=0.92)
    ax1.set_xticks(x); ax1.set_xticklabels(MODELS, fontsize=8.5, rotation=20)
    ax1.set_ylabel("Training Duration (Seconds)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Computational Cost: Training Duration on Joint Dataset", fontsize=10.5, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5, loc="upper left")

    # Panel (b): Training Overhead (% over Baseline): G0 vs B1 vs G1 vs B2 across all 7 models
    w2 = 0.18
    ov_g0 = []; ov_b1 = []; ov_g1 = []; ov_b2 = []

    for m in MODELS:
        t0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]["training_seconds"]
        tg0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]["training_seconds"]
        tb1 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]["training_seconds"]
        tg1 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G1")].iloc[0]["training_seconds"]
        tb2 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B2")].iloc[0]["training_seconds"]

        ov_g0.append((tg0 - t0) / t0 * 100 if t0 > 0 else 0)
        ov_b1.append((tb1 - t0) / t0 * 100 if t0 > 0 else 0)
        ov_g1.append((tg1 - t0) / t0 * 100 if t0 > 0 else 0)
        ov_b2.append((tb2 - t0) / t0 * 100 if t0 > 0 else 0)

    ax2.bar(x - 1.5*w2, ov_g0, w2, label="GenEDA 1-Stg (G0)", color="#F59E0B", alpha=0.92)
    ax2.bar(x - 0.5*w2, ov_b1, w2, label="CyberEDA 1-Stg (B1)", color="#10B981", alpha=0.92)
    ax2.bar(x + 0.5*w2, ov_g1, w2, label="GenEDA 2-Stg (G1)", color="#D97706", alpha=0.92)
    ax2.bar(x + 1.5*w2, ov_b2, w2, label="CyberEDA 2-Stg (B2)", color="#047857", alpha=0.92)

    ax2.set_xticks(x); ax2.set_xticklabels(MODELS, fontsize=8.5, rotation=20)
    ax2.set_ylabel("Overhead over Baseline (%)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Augmentation Training Overhead over Baseline (A0)", fontsize=10.5, fontweight="bold")
    max_ov = max(max(ov_g0), max(ov_b1), max(ov_g1), max(ov_b2))
    ax2.set_ylim(0, max(max_ov * 1.35, 230))
    ax2.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.0, loc="upper left")

    plt.suptitle("Figure 8: Computational Cost and Augmentation Training Overhead Across 7 Architectures on Joint Dataset",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.12)

    caption = (
        "Notes: Measures wall-clock training duration across 7 architectures under identical GPU hardware environment.\n"
        "Calculation: Training Overhead (%) = [(T_aug - T_base) / T_base] × 100%."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig09_topk_ranking_soc_metrics(df, output_path):
    """Figure 9: SOC Operational Retrieval Quality (Hit@3, Hit@5, MRR): Baseline vs GenericEDA vs CyberEDA."""
    df_sub = df[df["dataset"] == "joint"].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 5.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    x = np.arange(len(MODELS))
    w = 0.26

    # Panel (a): Hit@3 Comparison (Baseline A0 vs GenericEDA G0 vs CyberEDA B1)
    h3_a0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]["hit3_minf2"] * 100 for m in MODELS]
    h3_g0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]["hit3_minf2"] * 100 for m in MODELS]
    h3_b1 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]["hit3_minf2"] * 100 for m in MODELS]

    ax1.bar(x - w, h3_a0, w, label="Baseline (A0)", color="#64748B", alpha=0.92)
    ax1.bar(x, h3_g0, w, label="GenericEDA (G0)", color="#F59E0B", alpha=0.92)
    ax1.bar(x + w, h3_b1, w, label="CyberEDA (B1)", color="#10B981", alpha=0.92)
    ax1.set_xticks(x); ax1.set_xticklabels(MODELS, fontsize=8.5, rotation=20)
    ax1.set_ylabel("Hit@3 Score (%)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Multi-Label Incident Triage: Top-3 Retrieval (Hit@3)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(60, 96)
    ax1.legend(frameon=True, fontsize=8.5, loc="lower right")

    # Panel (b): Mean Reciprocal Rank (MRR)
    mrr_a0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]["mrr_minf2"] for m in MODELS]
    mrr_g0 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]["mrr_minf2"] for m in MODELS]
    mrr_b1 = [df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]["mrr_minf2"] for m in MODELS]

    ax2.bar(x - w, mrr_a0, w, label="Baseline (A0)", color="#64748B", alpha=0.92)
    ax2.bar(x, mrr_g0, w, label="GenericEDA (G0)", color="#F59E0B", alpha=0.92)
    ax2.bar(x + w, mrr_b1, w, label="CyberEDA (B1)", color="#10B981", alpha=0.92)
    ax2.set_xticks(x); ax2.set_xticklabels(MODELS, fontsize=8.5, rotation=20)
    ax2.set_ylabel("MRR Score", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Ranking Precision: Mean Reciprocal Rank (MRR)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0.65, 0.90)
    ax2.legend(frameon=True, fontsize=8.5, loc="lower right")

    plt.suptitle("Figure 9: Security Operations Center (SOC) Alert Triage and Retrieval Performance on Joint Dataset",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.12)

    caption = (
        "Notes: Baseline scenario (A0) evaluated under standard Global Threshold (τ_g); Augmentation scenarios (G0, B1) evaluated under Minority-F2 Threshold (τ_m).\n"
        "Evaluates multi-label ranking quality across 4,001 test reports."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig10_performance_cost_pareto_tradeoff(df, output_path):
    """Figure 10: Performance Gain vs. Computational Cost Trade-off (Two-Panel Grouped Bar Chart)."""
    df_sub = df[df["dataset"] == "joint"].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 5.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.45)

    x = np.arange(len(MODELS))
    w = 0.30

    d_f1_g0 = []; d_f1_b1 = []
    d_time_g0 = []; d_time_b1 = []

    for m in MODELS:
        r_a0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]
        r_g0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]
        r_b1 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]

        f0 = r_a0["macro_f1_global"] * 100; t0 = r_a0["training_seconds"]
        fg = r_g0["macro_f1_minf2"] * 100; tg = r_g0["training_seconds"]
        fb = r_b1["macro_f1_minf2"] * 100; tb = r_b1["training_seconds"]

        d_f1_g0.append(fg - f0)
        d_f1_b1.append(fb - f0)
        d_time_g0.append(tg - t0)
        d_time_b1.append(tb - t0)

    # Panel (a): Performance Gain (ΔMacro-F1 in percentage points pp)
    r1_a = ax1.bar(x - w/2, d_f1_g0, w, label="GenericEDA (G0 - A0)", color="#F59E0B", alpha=0.92)
    r2_a = ax1.bar(x + w/2, d_f1_b1, w, label="CyberEDA (B1 - A0)", color="#10B981", alpha=0.92)

    for rect in list(r1_a) + list(r2_a):
        val = rect.get_height()
        va = "bottom" if val >= 0 else "top"
        offset_y = 0.25 if val >= 0 else -0.5
        ax1.text(rect.get_x() + rect.get_width()/2, val + offset_y, f"{val:+.1f} pp",
                 ha="center", va=va, fontsize=7.2, fontweight="bold", color="#1E293B")

    ax1.set_xticks(x)
    ax1.set_xticklabels(MODELS, fontsize=8.5, rotation=15)
    ax1.set_ylabel("Macro-F1 Gain over Baseline (pp)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Performance Gain: Macro-F1 Improvement over Baseline (A0)", fontsize=10.5, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax1.set_ylim(0, max(max(d_f1_b1), max(d_f1_g0)) * 1.25)

    # Panel (b): Computational Overhead (ΔTraining Time in seconds)
    r1_b = ax2.bar(x - w/2, d_time_g0, w, label="GenericEDA (G0 - A0)", color="#F59E0B", alpha=0.92)
    r2_b = ax2.bar(x + w/2, d_time_b1, w, label="CyberEDA (B1 - A0)", color="#10B981", alpha=0.92)

    for rect in list(r1_b) + list(r2_b):
        val = rect.get_height()
        va = "bottom" if val >= 0 else "top"
        offset_y = 15 if val >= 0 else -25
        ax2.text(rect.get_x() + rect.get_width()/2, val + offset_y, f"+{val:.0f}s",
                 ha="center", va=va, fontsize=7.2, fontweight="bold", color="#1E293B")

    ax2.set_xticks(x)
    ax2.set_xticklabels(MODELS, fontsize=8.5, rotation=15)
    ax2.set_ylabel("Additional Training Duration (Seconds)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Computational Cost: Additional Training Time over Baseline (A0)", fontsize=10.5, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax2.set_ylim(0, max(max(d_time_b1), max(d_time_g0)) * 1.20)

    plt.suptitle("Figure 10: Performance Gain vs. Computational Cost Trade-Off Across 7 Architectures on Joint Dataset",
                 fontsize=12.5, fontweight="bold", y=0.98)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.06, right=0.98, wspace=0.12)

    caption = (
        "Notes: Compares 1-Stage Augmentation scenarios (GenericEDA G0, CyberEDA B1) against Baseline (A0).\n"
        "Calculation: ΔMacro-F1 = Macro-F1(Aug) - Macro-F1(A0) (pp); ΔTraining Time = Training Time(Aug) - Training Time(A0) (seconds)."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


def generate_fig11_augmentation_efficiency_and_saturation(df, output_path):
    """Figure 11: Augmentation Efficiency Across 7 Architectures (1-Stage vs 2-Stage)."""
    df_sub = df[df["dataset"] == "joint"].copy()
    fig, ax = plt.subplots(figsize=(14.5, 6.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    x = np.arange(len(MODELS))
    w = 0.19
    eff_b1 = []; eff_g0 = []; eff_b2 = []; eff_g1 = []

    for m in MODELS:
        r_a0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "A0")].iloc[0]
        r_b1 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B1")].iloc[0]
        r_g0 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G0")].iloc[0]
        r_b2 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "B2")].iloc[0]
        r_g1 = df_sub[(df_sub["model"] == m) & (df_sub["scenario"] == "G1")].iloc[0]

        t0 = r_a0["training_seconds"]; f0 = r_a0["macro_f1_global"] * 100
        tb1 = r_b1["training_seconds"]; fb1 = r_b1["macro_f1_minf2"] * 100
        tg0 = r_g0["training_seconds"]; fg0 = r_g0["macro_f1_minf2"] * 100
        tb2 = r_b2["training_seconds"]; fb2 = r_b2["macro_f1_minf2"] * 100
        tg1 = r_g1["training_seconds"]; fg1 = r_g1["macro_f1_minf2"] * 100

        e_b1 = ((fb1 - f0) / (tb1 - t0) * 1000) if (tb1 - t0) > 0 else 0
        e_g0 = ((fg0 - f0) / (tg0 - t0) * 1000) if (tg0 - t0) > 0 else 0
        e_b2 = ((fb2 - f0) / (tb2 - t0) * 1000) if (tb2 - t0) > 0 else 0
        e_g1 = ((fg1 - f0) / (tg1 - t0) * 1000) if (tg1 - t0) > 0 else 0

        eff_b1.append(e_b1)
        eff_g0.append(e_g0)
        eff_b2.append(e_b2)
        eff_g1.append(e_g1)

    r1 = ax.bar(x - 1.5*w, eff_b1, w, label="CyberEDA 1-Stage (B1)", color="#10B981", alpha=0.92)
    r2 = ax.bar(x - 0.5*w, eff_g0, w, label="GenericEDA 1-Stage (G0)", color="#F59E0B", alpha=0.92)
    r3 = ax.bar(x + 0.5*w, eff_b2, w, label="CyberEDA 2-Stage (B2)", color="#047857", alpha=0.92)
    r4 = ax.bar(x + 1.5*w, eff_g1, w, label="GenericEDA 2-Stage (G1)", color="#D97706", alpha=0.92)

    for rect in list(r1) + list(r2) + list(r3) + list(r4):
        val = rect.get_height()
        if val > 0.5:
            ax.text(rect.get_x() + rect.get_width()/2, val + 0.8, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1E293B")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9.5, fontweight="bold", rotation=15)
    ax.set_ylabel("Augmentation Efficiency (ΔMacro-F1 / ΔTime × 1,000)", fontsize=10.5, fontweight="bold")
    ax.set_title("Figure 11: Augmentation Efficiency Across 7 Architectures on Joint Dataset (1-Stage vs. 2-Stage)",
                 fontsize=12.5, fontweight="bold", pad=20)
    ax.legend(frameon=True, fontsize=8.8, loc="upper right", facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax.set_ylim(0, max(max(eff_b1), max(eff_g0)) * 1.20)

    fig.subplots_adjust(bottom=0.20, top=0.90, left=0.08, right=0.95)

    caption = (
        "Notes: Augmentation scenarios (B1, B2, G0, G1) evaluated under Minority-F2 Threshold (τ_m); Baseline scenario (A0) evaluated under standard Global Threshold (τ_g).\n"
        "Calculation: Efficiency = [(Macro-F1_aug - Macro-F1_base) / (T_aug - T_base)] × 1,000."
    )
    fig.text(0.5, 0.04, caption, ha="center", va="bottom", fontsize=8.2, color="#334155", linespacing=1.35)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"[OK] Figure saved: {output_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN EXECUTION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  TASK 7: ADVANCED BENCHMARK ANALYSIS & PUBLICATION ASSET GENERATION")
    print(f"  Destination Folder: {FIG_DIR}")
    print("=" * 80)

    # Clean old figure and table PNGs in destination directory
    for old_f in FIG_DIR.glob("*.png"):
        try:
            old_f.unlink()
        except Exception:
            pass

    df = load_all_benchmark_data()
    if len(df) == 0:
        print("[FATAL] No data found.")
        sys.exit(1)

    print("\n--- Generating 7 Publication Table Images (Side-by-Side Global & Min-F2) ---")
    generate_table01_main_benchmark_dataset(df, "joint", FIG_DIR / "table01_main_benchmark_joint.png")
    generate_table01_main_benchmark_dataset(df, "cti-mitre", FIG_DIR / "table01_main_benchmark_cti.png")
    generate_table01_main_benchmark_dataset(df, "tram", FIG_DIR / "table01_main_benchmark_tram.png")
    generate_table02_thresholding_comparison(df, FIG_DIR / "table02_thresholding_direct_comparison.png")
    generate_table03_training_overhead(df, FIG_DIR / "table03_training_overhead_efficiency.png")
    generate_table04_frequency_tier(df, FIG_DIR / "table04_frequency_tier_breakdown.png")
    generate_table05_retrieval_ranking(df, FIG_DIR / "table05_retrieval_ranking.png")

    print("\n--- Generating 11 Scientific Figures with Clear Top Titles & Bottom Notes ---")
    generate_fig01_macro_f1_benchmark_minf2(df, FIG_DIR / "fig01_macro_f1_benchmark_minority_f2.png")
    generate_fig02_tier_breakdown_comparison(df, FIG_DIR / "fig02_3tier_frequency_breakdown_comparison.png")
    generate_fig03_macro_micro_gap(df, FIG_DIR / "fig03_macro_vs_micro_imbalance_gap.png")
    generate_fig04_cyber_vs_generic_head_to_head(df, FIG_DIR / "fig04_cyber_vs_generic_head_to_head.png")
    generate_fig05_zf1_reduction(df, FIG_DIR / "fig05_zf1_reduction_baseline_generic_cyber.png")
    generate_fig06_recall_and_precision_tradeoff(df, FIG_DIR / "fig06_recall_and_precision_tradeoff.png")
    generate_fig07_macro_f1_benchmark_global(df, FIG_DIR / "fig07_macro_f1_benchmark_global_threshold.png")
    generate_fig08_training_time_and_overhead(df, FIG_DIR / "fig08_training_time_and_overhead_barchart.png")
    generate_fig09_topk_ranking_soc_metrics(df, FIG_DIR / "fig09_topk_ranking_soc_metrics.png")
    generate_fig10_performance_cost_pareto_tradeoff(df, FIG_DIR / "fig10_performance_cost_pareto_tradeoff.png")
    generate_fig11_augmentation_efficiency_and_saturation(df, FIG_DIR / "fig11_augmentation_efficiency_and_saturation.png")

    print("\n" + "=" * 80)
    print(f"  [ALL DONE] Successfully generated 7 Table Images & 11 Figures in:")
    print(f"  {FIG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
