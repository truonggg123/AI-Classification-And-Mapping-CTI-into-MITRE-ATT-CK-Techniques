import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

BASE_DIR = Path(r"d:\Truong\FPT\SUMMER2026\AIC211\CTI_ATT&CK\new project\AI-Classification-And-Mapping-CTI-into-MITRE-ATT-CK-Techniques")
RESULTS_DIR = BASE_DIR / "results" / "DistilBERT_MultiSeed"
OUTPUT_REPORT = BASE_DIR / "analysis" / "multiseed_stability_report.txt"
OUTPUT_LATEX = BASE_DIR / "analysis" / "multiseed_stability_table.tex"

SEEDS = [42, 123, 456, 789, 2026]

def scan_results():
    records = []
    if not RESULTS_DIR.exists():
        print(f"[WARN] Results directory {RESULTS_DIR} does not exist yet.")
        return pd.DataFrame()

    for seed in SEEDS:
        seed_dir = RESULTS_DIR / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        for exp_dir in sorted(seed_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            # Try seed_summary.json, then metrics.json, then multi_seed_record.csv
            summary_file = exp_dir / "seed_summary.json"
            metrics_file = exp_dir / "metrics.json"
            csv_file = exp_dir / "multi_seed_record.csv"

            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(data)
            elif csv_file.exists():
                df_temp = pd.read_csv(csv_file)
                records.extend(df_temp.to_dict(orient="records"))
            elif metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                rec = {
                    "seed": m.get("seed", seed),
                    "scenario": m.get("scenario", exp_dir.name.split("_")[0]),
                    "dataset": exp_dir.name.split("_")[-1] if "_" in exp_dir.name else "joint",
                    "macro_f1": m.get("test_global", {}).get("macro_f1", 0.0),
                    "micro_f1": m.get("test_global", {}).get("micro_f1", 0.0),
                    "weighted_f1": m.get("test_global", {}).get("weighted_f1", 0.0),
                    "minority_f2": m.get("test_minority_macro_f2", 0.0),
                    "minority_recall": m.get("test_minority_macro_recall", 0.0),
                    "zero_f1_count": m.get("test_global", {}).get("zero_f1_count", 0),
                    "training_seconds": m.get("training_seconds", 0.0),
                }
                records.append(rec)
    
    return pd.DataFrame(records)

def generate_report(df):
    lines = []
    lines.append("=" * 88)
    lines.append("TASK 8: MULTI-SEED STABILITY ANALYSIS REPORT — DISTILBERT (B1 vs G0)")
    lines.append("=" * 88)
    lines.append(f"Target Seeds: {SEEDS}")
    lines.append(f"Total Runs Found: {len(df)}")
    lines.append("")

    if df.empty:
        lines.append("No multi-seed runs found yet in results/DistilBERT_MultiSeed/.")
        lines.append("Please execute the notebooks on Kaggle/Colab or locally, then run this script.")
        content = "\n".join(lines)
        OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_REPORT.write_text(content, encoding="utf-8")
        print(content)
        return

    # Filter scenarios B1 and G0
    for dataset in df["dataset"].unique():
        lines.append("-" * 88)
        lines.append(f"DATASET SUBSET: {dataset.upper()}")
        lines.append("-" * 88)

        df_ds = df[df["dataset"] == dataset]
        
        # Individual Runs Table
        lines.append("\n1. INDIVIDUAL RUNS BREAKDOWN:")
        lines.append(f"{'Seed':<8} | {'Scenario':<8} | {'Macro F1':<12} | {'Micro F1':<12} | {'Minority F2':<12} | {'Zero-F1':<8} | {'Time (s)':<10}")
        lines.append("-" * 80)
        for _, r in df_ds.sort_values(by=["scenario", "seed"]).iterrows():
            mf1 = f"{r.get('macro_f1', 0)*100:.2f}%"
            micf1 = f"{r.get('micro_f1', 0)*100:.2f}%"
            f2 = f"{r.get('minority_f2', 0)*100:.2f}%" if pd.notna(r.get('minority_f2')) else "N/A"
            zf1 = f"{int(r.get('zero_f1_count', 0))}"
            ts = f"{r.get('training_seconds', 0):.1f}s"
            lines.append(f"{r['seed']:<8} | {r['scenario']:<8} | {mf1:<12} | {micf1:<12} | {f2:<12} | {zf1:<8} | {ts:<10}")

        # Summary Mean +- Std Table
        lines.append("\n2. STATISTICAL STABILITY (MEAN ± STD across seeds):")
        lines.append(f"{'Metric':<25} | {'GenericEDA (G0)':<22} | {'CyberEDA (B1)':<22} | {'Δ (B1 - G0)':<12} | {'p-value':<10}")
        lines.append("-" * 98)

        b1_sub = df_ds[df_ds["scenario"] == "B1"].sort_values(by="seed")
        g0_sub = df_ds[df_ds["scenario"] == "G0"].sort_values(by="seed")

        metrics_to_eval = [
            ("Macro F1 (%)", "macro_f1", 100),
            ("Micro F1 (%)", "micro_f1", 100),
            ("Minority Macro-F2 (%)", "minority_f2", 100),
            ("Minority Recall (%)", "minority_recall", 100),
            ("Zero-F1 Count", "zero_f1_count", 1),
            ("Rare F1 (<30) (%)", "rare_f1", 100),
            ("Training Time (s)", "training_seconds", 1),
        ]

        for label, col, scale in metrics_to_eval:
            if col in b1_sub.columns and col in g0_sub.columns:
                b1_vals = b1_sub[col].dropna() * scale
                g0_vals = g0_sub[col].dropna() * scale

                b1_m, b1_s = b1_vals.mean(), b1_vals.std()
                g0_m, g0_s = g0_vals.mean(), g0_vals.std()
                diff = b1_m - g0_m

                p_val_str = "N/A"
                if HAS_SCIPY and len(b1_vals) == len(g0_vals) and len(b1_vals) >= 3:
                    try:
                        _, p_val = stats.ttest_rel(b1_vals, g0_vals)
                        p_val_str = f"{p_val:.4f}" + ("*" if p_val < 0.05 else "")
                    except Exception:
                        pass

                lines.append(f"{label:<25} | {g0_m:>8.2f} ± {g0_s:<8.2f}   | {b1_m:>8.2f} ± {b1_s:<8.2f}   | {diff:>+10.2f}   | {p_val_str:<10}")

    content = "\n".join(lines)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(content, encoding="utf-8")
    print(content)
    print(f"\n[DONE] Saved report to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    df = scan_results()
    generate_report(df)
