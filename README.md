# AI-Classification-And-Mapping-CTI-into-MITRE-ATT-CK-Techniques

Automated mapping of Cyber Threat Intelligence (CTI) reports to **MITRE ATT&CK techniques** using domain-aware text augmentation (**CyberEDA**) and deep learning models.

---

## 🎯 1. Objective & Datasets

- **Goal:** Automatically classify raw threat intelligence text into MITRE ATT&CK techniques ($Txxxx$) while overcoming severe **class imbalance** (especially rare, underrepresented "tail classes").
- **Datasets** (raw data stored in `dataset/raw/`):
  - **`cti_to_mitre`**: 12,865 samples, 188 techniques (high imbalance real-world CTI).
  - **`tram`**: 7,249 samples, 50 techniques (curated benchmark from MITRE TRAM).
  - **`joint`**: 20,114 samples, 188 techniques (merged unified dataset).

---

## 🔬 2. Methodology Overview

1. **Data Augmentation (EDA with $\alpha = 0.10$):**
   - **GenericEDA:** Uses WordNet for Synonym Replacement (SR) and Random Insertion (RI).
   - **CyberEDA (Proposed):** Replaces entities strictly using aliases from STIX 2.1 & MITRE ATT&CK (threat actors, malware, tools) to prevent semantic distortion of technical indicators.
   - **Perturbations:** Combines SR, RI, Random Swap (RS), and Random Deletion (RD).
2. **Architectures Evaluated (7 Models):**
   - Baseline: `TextCNN`
   - General Transformers: `BERT`, `DistilBERT`, `ModernBERT`
   - Cybersecurity Domain Transformers: `SecBERT`, `SecureBERT`, `CySecBERT`
3. **Thresholding Strategy:**
   - **Minority-F2 Threshold:** Optimizes decision thresholds on minority classes to maximize detection recall on rare threat techniques.

---

## 🚀 3. Quick Start: How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 2: Build Preprocessed Data from Raw Files
Because only raw data is tracked on Git, run these two commands to generate cleaned, stratified 70/10/20 splits:
```bash
# 1. Merge raw datasets & remove direct label leakage
python src/merge_datasets.py --target_dataset all

# 2. Anonymize entities ([CVE], [IPV4], etc.) & create stratified splits
python src/preprocessing.py --target_dataset all
```
*Outputs are created in `dataset/processed/<dataset_name>/` (`train.csv`, `val.csv`, `test.csv`).*

### Step 3: Generate Augmented Datasets (CyberEDA & GenericEDA)
Generate balanced augmentation training sets for scenarios B1/B2 and G0/G1:
```bash
# 1. Generate CyberEDA training set (STIX 2.1 Aliases) -> train_augmented_eda.csv
python src/cyber_eda.py --target_dataset all --mode eda

# 2. Generate GenericEDA training set (WordNet) -> train_augmented_generic_eda.csv
python src/generic_eda.py --target_dataset all
```
*Outputs are created in `dataset/processed/<dataset_name>/` (`train_augmented_eda.csv`, `train_augmented_generic_eda.csv`).*

### Step 4: Train Models
Run the training notebooks located in `notebooks/` (e.g., `notebooks/DistilBert notebooks/`).

### Step 5: Generate Publication Figures & Tables
```bash
# Generate main benchmark figures (Fig 1-11, Table 1-5)
python analysis/analyze_main_experiments.py

# Multi-seed stability & paired t-test analysis
python analysis/analyze_multiseed_stability.py

# Human audit label preservation analysis
python analysis/label_preservation/analyze_preservation_results.py
```

---

## 📊 4. Where to Find Results & Figures

All publication-ready charts and tables (300 DPI) are saved in `results/figures/`:

| Directory | What to Look For |
| :--- | :--- |
| **`results/figures/main_experiments/`** | • `fig01_macro_f1_benchmark_minority_f2.png`: 7-model benchmark<br>• `fig02_3tier_frequency_breakdown_comparison.png`: Tail vs Med vs Head gains<br>• `fig08_training_time_and_overhead_barchart.png`: Training overhead<br>• `fig10_performance_cost_pareto_tradeoff.png`: Pareto speed vs F1 trade-off<br>• `table01_main_benchmark_*.png`: Full numerical tables |
| **`results/figures/label_preservation/`** | • `table_label_preservation_summary.png`: Double-blind human audit results<br>• `fig_eda_operations_risk_analysis.png`: Relative Risk ($RR$) per EDA operation |
| **`results/figures/multiseed_stability/`** | • `fig01_multiseed_macro_f1_with_error_bars.png`: 5-seed stability test<br>• `table01_multiseed_overall_benchmark.png`: Paired t-test ($p > 0.48$) |
| **`results/figures/eda_ablation/`** | • Ratio tuning ($\alpha \in [0.02, 0.30]$) & operation ablation (SR, RI, RS, RD) |

### Key Experimental Takeaways:
- **Tail Class Rescue:** Augmentation boosts minority class F1 by **+14 to +23 pp** and slashes missed techniques (Zero-F1) from ~40 down to **7–15**.
- **Safer Augmentation:** CyberEDA achieves **82.0% label preservation** compared to 74.0% for GenericEDA (+8.0% safer).
- **Best Efficiency:** `DistilBERT` matches domain models in performance (**63.24% Macro-F1**) while training **2–3$\times$ faster** (~11 mins).

---

## 📁 5. Project Directory Structure

```
.
├── dataset/
│   ├── raw/                 # Original untouched datasets (dataset.csv, TRAM json)
│   └── processed/           # Generated train/val/test splits (created by step 2)
├── src/                     # Core pipeline modules
│   ├── merge_datasets.py    # Merging raw sources & neutralizing label leakage
│   ├── preprocessing.py     # Anonymization & multilabel stratified splitting
│   ├── cyber_eda.py         # STIX-based domain data augmentation
│   └── generic_eda.py       # WordNet-based generic data augmentation
├── notebooks/               # Training notebooks organized by architecture
├── analysis/                # Statistical testing and publication plotting scripts
├── results/                 # Numerical metrics.json and exported figure assets
│   └── figures/             # High-resolution benchmark figures and tables
├── requirements.txt         # Clean dependencies list
└── README.md
```
