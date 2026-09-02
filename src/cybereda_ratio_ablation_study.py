"""
Cyber EDA Ratio Tuning Study — CTI-to-MITRE Benchmark
=====================================================
Systematic Hyperparameter Tuning for Uniform EDA Augmentation Intensity alpha in [0.01, 0.20]
with Step Size = 0.01 (21 Configurations: 1 Baseline + 20 Uniform Ratios).

Theoretical Foundation:
----------------------
Based on the seminal paper:
Wei, J., & Zou, K. (2019). "EDA: Easy Data Augmentation Techniques for Boosting
Performance on Text Classification Tasks." EMNLP 2019.

The authors established uniform ratio scaling across all 4 operations:
alpha_sr = alpha_ri = alpha_rs = p_rd = alpha
with an empirical sweet-spot benchmark centered around alpha = 0.10.

Evaluation Protocol:
- Selection Criterion: Optimal alpha* selected strictly on validation split (val.csv, 10%).
- Final Reporting: Evaluated on held-out test split (test.csv, 20%).
- Proxy Model: OneVsRest LinearSVC + Hybrid TF-IDF (Word (1,2) + Char (2,4)).
- Output Folder: results/cybereda_ratio_ablation/
"""

import os, sys, re, json, time, pickle, random, argparse, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy.sparse import hstack

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from src.augmentation import build_cyber_knowledge_base, cyber_eda


# ===========================================================================
# Metrics & Utilities
# ===========================================================================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)


def compute_precision_recall_f1(y_true, y_pred):
    return {
        "Precision_Macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "Recall_Macro":    float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "Macro_F1":        float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "Precision_Micro": float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        "Recall_Micro":    float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        "Micro_F1":        float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
    }


def analyze_4_tier(y_true, y_pred, label_counts, labels_list):
    f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per  = recall_score(y_true, y_pred, average=None, zero_division=0)
    tier_defs = {
        "Head (>500)":     lambda c: c >= 500,
        "Major (100-499)": lambda c: 100 <= c < 500,
        "Medium (30-99)":  lambda c: 30 <= c < 100,
        "Tail (<30)":      lambda c: c < 30,
    }
    tier_summary   = {}
    zero_f1_labels = [lbl for i, lbl in enumerate(labels_list) if f1_per[i] == 0.0]
    for name, fn in tier_defs.items():
        idx = [i for i, lbl in enumerate(labels_list) if fn(label_counts.get(lbl, 0))]
        if idx:
            tier_summary[name] = {
                "num_labels": len(idx),
                "precision":  float(np.mean(prec_per[idx])),
                "recall":     float(np.mean(rec_per[idx])),
                "f1_score":   float(np.mean(f1_per[idx])),
                "zero_f1_count": int(sum(1 for i in idx if f1_per[i] == 0.0)),
            }
        else:
            tier_summary[name] = {"num_labels": 0, "precision": 0.0,
                                  "recall": 0.0, "f1_score": 0.0, "zero_f1_count": 0}
    return tier_summary, zero_f1_labels


def generate_augmented_dataset(df_train, protected_set, synonym_dict,
                                target_count=48, alpha_sr=0.05, alpha_ri=0.05,
                                alpha_rs=0.05, p_rd=0.05, seed=42):
    set_seed(seed)
    df = df_train.copy()
    if 'is_augmented' not in df.columns:
        df['is_augmented'] = 0
    df['_lbl'] = df['Labels'].apply(lambda x: str(x).split(','))
    lc = Counter(lbl for sub in df['_lbl'] for lbl in sub)
    minority = {lbl: c for lbl, c in lc.items() if c < target_count}
    l2i = {}
    for loc_i, row in df.iterrows():
        for lbl in row['_lbl']:
            l2i.setdefault(lbl, []).append(loc_i)
    rows_to_aug = []
    for lbl, count in minority.items():
        rows_to_aug.extend(random.choices(l2i[lbl], k=target_count - count))

    def _aug(ei):
        seq_i, loc_i = ei
        set_seed(seed + seq_i)
        row = df.loc[loc_i]
        aug = cyber_eda(str(row['Cleaned_Text']), protected_set, synonym_dict,
                        alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd)
        aug = re.sub(r'\s+', ' ', str(aug)).strip()
        toks = re.findall(r"[a-z0-9_\[\]]+(?:[./:-][a-z0-9_\[\]]+)*", aug.lower())
        return {'Cleaned_Text': aug, 'Labels': row['Labels'],
                'Label_Count': row['Label_Count'], 'Tokenized_Text': " ".join(toks),
                'source_sample_id': row['source_sample_id'], 'is_augmented': 1}

    with ThreadPoolExecutor(max_workers=8) as ex:
        records = list(ex.map(_aug, enumerate(rows_to_aug)))
    return pd.concat([df.drop(columns=['_lbl']), pd.DataFrame(records)], ignore_index=True)


# ===========================================================================
# Visualization Functions (Academic Quality)
# ===========================================================================

def plot_macro_f1_comparison(results, out_dir, dataset_name="cti_to_mitre"):
    ids    = [r['config_id'] for r in results]
    labels = [f"alpha={r['ratios']['alpha_sr']:.2f}" if r['config_id'] != 'Config_0' else 'Baseline' for r in results]
    val_f1 = [r['val_metrics']['Macro_F1']  for r in results]
    tst_f1 = [r['test_metrics']['Macro_F1'] for r in results]
    
    x = np.arange(len(ids)); w = 0.38
    fig, ax = plt.subplots(figsize=(14, 5.5))
    
    best_idx = np.argmax(val_f1[1:]) + 1
    val_colors = ['#78909C'] + ['#1976D2' if i != best_idx else '#0D47A1' for i in range(1, len(results))]
    test_colors = ['#90A4AE'] + ['#42A5F5' if i != best_idx else '#1565C0' for i in range(1, len(results))]
    
    ax.bar(x - w/2, val_f1, w, color=val_colors, alpha=0.90, label='Validation Macro F1')
    ax.bar(x + w/2, tst_f1, w, color=test_colors, alpha=0.55, hatch='//', label='Held-out Test Macro F1')
    
    bv, bt = results[0]['val_metrics']['Macro_F1'], results[0]['test_metrics']['Macro_F1']
    ax.axhline(bv, color='#37474F', ls='--', lw=1.2, label=f'Baseline Val F1 ({bv:.4f})')
    ax.axhline(bt, color='#455A64', ls=':',  lw=1.2, label=f'Baseline Test F1 ({bt:.4f})')
    
    for xi, (v, t) in enumerate(zip(val_f1, tst_f1)):
        ax.text(xi-w/2, v+0.0008, f'{v:.4f}', ha='center', va='bottom', fontsize=7.2, rotation=90)
        ax.text(xi+w/2, t+0.0008, f'{t:.4f}', ha='center', va='bottom', fontsize=7.2, rotation=90)
        
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Macro F1 Score', fontsize=11)
    ax.set_ylim(min(val_f1+tst_f1)-0.025, max(val_f1+tst_f1)+0.045)
    ax.set_title(f'Figure 1: Cyber EDA Uniform Ratio Tuning on {dataset_name.upper()} (alpha in [0.01, 0.20], step=0.01)\n'
                 'Dark Blue: Optimal Validation Config (alpha*) | Solid: Validation | Hatched: Test', fontsize=11)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    p = out_dir / 'figure_01_macro_f1_comparison.png'
    plt.savefig(p, dpi=150); plt.close(); print(f"  [SAVED] {p.name}")


def plot_4tier_breakdown(results, out_dir, dataset_name="cti_to_mitre"):
    labels = [f"alpha={r['ratios']['alpha_sr']:.2f}" if r['config_id'] != 'Config_0' else 'Baseline' for r in results]
    major  = [r['val_tier_summary']['Major (100-499)']['f1_score'] for r in results]
    medium = [r['val_tier_summary']['Medium (30-99)']['f1_score']  for r in results]
    tail   = [r['val_tier_summary']['Tail (<30)']['f1_score']      for r in results]
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.bar(x-w,  major,  w, label='Major (100-499) F1', color='#1565C0', alpha=0.85)
    ax.bar(x,    medium, w, label='Medium (30-99) F1',  color='#E65100', alpha=0.85)
    ax.bar(x+w,  tail,   w, label='Tail (<30) F1',      color='#C62828', alpha=0.85)
    for xi, t in enumerate(tail):
        ax.text(xi+w, t+0.001, f'{t:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score (Validation Set)', fontsize=11)
    ax.set_title(f'Figure 2: 4-Tier Frequency Breakdown across Tuning Spectrum on {dataset_name.upper()} (Validation Set)\n'
                 'Tail F1 (<30 samples) measures rare MITRE ATT&CK technique recovery capability', fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    p = out_dir / 'figure_02_4tier_breakdown.png'
    plt.savefig(p, dpi=150); plt.close(); print(f"  [SAVED] {p.name}")


def print_console_table(results):
    baseline = results[0]
    bv = baseline['val_metrics']['Macro_F1']
    bt = baseline['test_metrics']['Macro_F1']
    aug_sorted = sorted(results[1:], key=lambda r: r['val_metrics']['Macro_F1'], reverse=True)
    line = "=" * 98
    hdr  = f"  {'Config':<12} {'Alpha (alpha)':<14} {'Val Macro F1':>14} {'Test Macro F1':>14} {'Delta Val':>12} {'Delta Test':>12} {'ZF1-Test':>10}"
    sep  = "-" * 98
    print(f"\n{line}")
    print("  RANKED UNIFORM TUNING RESULTS (Sorted by Validation Macro F1)")
    print(hdr); print(sep)
    print(f"  {'Config_0':<12} {'0.00 (Base)':<14} {bv:>14.4f} {bt:>14.4f} {'---':>12} {'---':>12} {baseline['num_zero_f1_test_classes']:>10}  [BASELINE]")
    print(sep)
    for rank, r in enumerate(aug_sorted, 1):
        rat = r['ratios']
        rs  = f"{rat['alpha_sr']:.2f}"
        dv  = r['val_metrics']['Macro_F1']  - bv
        dt  = r['test_metrics']['Macro_F1'] - bt
        zf1 = r['num_zero_f1_test_classes']
        mark = " [OPTIMAL]" if rank == 1 else ""
        print(f"  {r['config_id']:<12} {rs:<14} {r['val_metrics']['Macro_F1']:>14.4f} {r['test_metrics']['Macro_F1']:>14.4f} {dv:>+12.4f} {dt:>+12.4f} {zf1:>10}{mark}")
    print(line)


# ===========================================================================
# Main Tuning Execution
# ===========================================================================

def run_ablation_study(target_dataset='joint'):
    project_root = Path(__file__).resolve().parent.parent
    data_dir     = project_root / 'dataset' / 'processed' / target_dataset
    out_dir      = project_root / 'results' / 'cybereda_ratio_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  CYBER EDA UNIFORM RATIO TUNING — BENCHMARK ({target_dataset.upper()})")
    print("  Theoretical Basis: Wei & Zou (EMNLP 2019) Uniform Alpha Paradigm")
    print("  Tuning Range: alpha in [0.01, 0.30] with Step Size = 0.01 (30 Ratios + Baseline)")
    print("=" * 80)

    # ── Step 1: Load pre-built splits ──────────────────────────────────
    print(f"\n[STEP 1] Loading dataset splits from: {data_dir}...")
    for f in [data_dir/'train.csv', data_dir/'val.csv', data_dir/'test.csv', data_dir/'multilabel_binarizer.pkl']:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}\n  -> Run: python src/preprocessing.py --target_dataset {target_dataset} first")

    df_train = pd.read_csv(data_dir / 'train.csv')
    df_val   = pd.read_csv(data_dir / 'val.csv')
    df_test  = pd.read_csv(data_dir / 'test.csv')

    with open(data_dir / 'multilabel_binarizer.pkl', 'rb') as fh:
        mlb = pickle.load(fh)

    labels_list = list(mlb.classes_)
    y_val  = mlb.transform(df_val['Labels'].apply(lambda x: str(x).split(',')))
    y_test = mlb.transform(df_test['Labels'].apply(lambda x: str(x).split(',')))

    all_labels = [lbl for sub in df_train['Labels'].apply(lambda x: str(x).split(',')) for lbl in sub]
    train_label_counts = Counter(all_labels)
    target_count = int(round(np.mean(list(train_label_counts.values()))))

    print(f"  Dataset: {target_dataset} | Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,} | Classes: {len(labels_list)} | Target: {target_count}/class")

    # ── Step 2: Build Cyber Knowledge Base ─────────────────────────────
    print("\n[STEP 2] Loading Cyber Knowledge Base (MITRE STIX)...")
    cache_dir = project_root / 'dataset' / 'processed'
    protected_set, synonym_dict = build_cyber_knowledge_base(cache_dir=cache_dir)

    # ── Step 3: Define 31 Uniform Configs (alpha in [0.01, 0.30]) ──────────
    configs = [
        {
            "id": "Config_0", "axis": "Baseline",
            "name": "No Augmentation",
            "description": "Original unaugmented training data lower-bound baseline.",
            "alpha_sr": 0.0, "alpha_ri": 0.0, "alpha_rs": 0.0, "p_rd": 0.0,
        }
    ]

    alpha_values = [round(a, 2) for a in np.arange(0.01, 0.301, 0.01)]
    for i, a in enumerate(alpha_values, 1):
        configs.append({
            "id": f"Config_{i}",
            "axis": "Uniform EDA Tuning",
            "name": f"Uniform Alpha={a:.2f}",
            "description": f"Uniform intensity alpha={a:.2f} across SR, RI, RS, and RD.",
            "alpha_sr": a, "alpha_ri": a, "alpha_rs": a, "p_rd": a,
        })

    # ── Step 4: Execute Tuning Grid ────────────────────────────────────
    print(f"\n[STEP 3] Executing {len(configs)} Configurations...\n")
    all_results = []

    for cfg in configs:
        cid = cfg["id"]
        sr, ri, rs, rd = cfg["alpha_sr"], cfg["alpha_ri"], cfg["alpha_rs"], cfg["p_rd"]
        t0 = time.time()

        if cid == "Config_0":
            df_aug  = df_train.copy()
            if 'is_augmented' not in df_aug.columns:
                df_aug['is_augmented'] = 0
            n_added = 0
        else:
            df_aug  = generate_augmented_dataset(
                df_train, protected_set, synonym_dict,
                target_count=target_count,
                alpha_sr=sr, alpha_ri=ri, alpha_rs=rs, p_rd=rd, seed=42)
            n_added = len(df_aug) - len(df_train)

        # Hybrid TF-IDF features
        wv = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
        cv = TfidfVectorizer(ngram_range=(2, 4), analyzer='char', max_features=25000, sublinear_tf=True)

        X_tr = hstack([wv.fit_transform(df_aug['Tokenized_Text'].fillna('')),
                       cv.fit_transform(df_aug['Cleaned_Text'].fillna(''))]).tocsr()
        X_va = hstack([wv.transform(df_val['Tokenized_Text'].fillna('')),
                       cv.transform(df_val['Cleaned_Text'].fillna(''))]).tocsr()
        X_te = hstack([wv.transform(df_test['Tokenized_Text'].fillna('')),
                       cv.transform(df_test['Cleaned_Text'].fillna(''))]).tocsr()

        y_aug = mlb.transform(df_aug['Labels'].apply(lambda x: str(x).split(',')))

        clf = OneVsRestClassifier(LinearSVC(class_weight='balanced', max_iter=1000, random_state=42))
        clf.fit(X_tr, y_aug)

        vp = (clf.decision_function(X_va) > 0).astype(int)
        tp = (clf.decision_function(X_te) > 0).astype(int)

        vm = compute_precision_recall_f1(y_val,  vp)
        tm = compute_precision_recall_f1(y_test, tp)
        vt, zv = analyze_4_tier(y_val,  vp, train_label_counts, labels_list)
        tt, zt = analyze_4_tier(y_test, tp, train_label_counts, labels_list)

        elapsed = time.time() - t0

        print(f"  [{cid:<9}] alpha={sr:.2f} | Val F1={vm['Macro_F1']:.4f} | Test F1={tm['Macro_F1']:.4f} | ZF1-Test={len(zt):>2} | {elapsed:.1f}s")

        all_results.append({
            "config_id":                cid,
            "axis":                     cfg["axis"],
            "name":                     cfg["name"],
            "description":              cfg["description"],
            "ratios":                   {"alpha_sr": sr, "alpha_ri": ri, "alpha_rs": rs, "p_rd": rd},
            "num_train_samples":        len(df_aug),
            "num_augmented_added":      n_added,
            "execution_time_seconds":   round(elapsed, 2),
            "val_metrics":              vm,
            "val_tier_summary":         vt,
            "num_zero_f1_val_classes":  len(zv),
            "test_metrics":             tm,
            "test_tier_summary":        tt,
            "num_zero_f1_test_classes": len(zt),
        })

    # ── Step 5: Summary & Winner Selection ──────────────────────────────
    print_console_table(all_results)

    aug_only  = [r for r in all_results if r['config_id'] != 'Config_0']
    baseline  = all_results[0]
    bv        = baseline['val_metrics']['Macro_F1']
    bt        = baseline['test_metrics']['Macro_F1']
    best_val  = max(aug_only, key=lambda r: r['val_metrics']['Macro_F1'])
    optimal_alpha = best_val['ratios']['alpha_sr']

    print(f"\n[OPTIMAL HYPERPARAMETER SELECTION]")
    print(f"  Selected Winner: {best_val['config_id']} (alpha* = {optimal_alpha:.2f})")
    print(f"  Validation Macro F1 = {best_val['val_metrics']['Macro_F1']:.4f} (Delta: {best_val['val_metrics']['Macro_F1']-bv:+.4f})")
    print(f"  Held-out Test Macro F1 = {best_val['test_metrics']['Macro_F1']:.4f} (Delta: {best_val['test_metrics']['Macro_F1']-bt:+.4f})")

    # ── Step 6: CSV Summary Export ──────────────────────────────────────
    rows = []
    for r in all_results:
        rat = r['ratios']
        vm, tm, vt = r['val_metrics'], r['test_metrics'], r['val_tier_summary']
        rows.append({
            "Config ID":     r['config_id'],
            "Alpha":         rat['alpha_sr'],
            "Val Macro F1":  round(vm['Macro_F1'], 4),
            "Val Major F1":  round(vt['Major (100-499)']['f1_score'], 4),
            "Val Medium F1": round(vt['Medium (30-99)']['f1_score'], 4),
            "Val Tail F1":   round(vt['Tail (<30)']['f1_score'], 4),
            "Val ZeroF1":    r['num_zero_f1_val_classes'],
            "Test Macro F1": round(tm['Macro_F1'], 4),
            "Test ZeroF1":   r['num_zero_f1_test_classes'],
            "Delta Val F1":  round(vm['Macro_F1']  - bv, 4),
            "Delta Test F1": round(tm['Macro_F1'] - bt, 4),
            "Time (s)":      r['execution_time_seconds'],
        })
    csv_name = f'cybereda_ablation_results_{target_dataset}.csv'
    pd.DataFrame(rows).to_csv(out_dir / csv_name, index=False, encoding='utf-8')
    print(f"\n  [SAVED] {out_dir / csv_name}")

    # ── Step 7: Render Academic Figures ─────────────────────────────────
    plot_macro_f1_comparison(all_results, out_dir, dataset_name=target_dataset)
    plot_4tier_breakdown(all_results, out_dir, dataset_name=target_dataset)

    print(f"\n{'=' * 80}")
    print(f"  ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")
    print(f"  OUTPUTS SAVED IN: results/cybereda_ratio_ablation/")
    print(f"  OPTIMAL alpha* = {optimal_alpha:.2f} (dataset: {target_dataset})")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cyber EDA Ratio Tuning Experiment")
    parser.add_argument('--target_dataset', type=str, default='joint',
                        choices=['cti_to_mitre', 'joint', 'tram'],
                        help='Dataset to run tuning on (default: joint)')
    args = parser.parse_args()
    run_ablation_study(target_dataset=args.target_dataset)
