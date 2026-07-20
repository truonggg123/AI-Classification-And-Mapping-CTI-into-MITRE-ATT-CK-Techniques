# 🛡️ CTI & MITRE ATT&CK Mapping System (Single-label Classification)

This repository contains the **Single-label Classification** implementation for mapping Cyber Threat Intelligence (CTI) reports and system logs to the MITRE ATT&CK framework at the **Parent Technique (`Txxxx`)** level.

## 🎯 Objectives
- **Goal:** Automatically classify CTI texts and logs, mapping them to MITRE ATT&CK **Parent Techniques**.
- **Approach:** Convert the multi-label problem into a **single-label** one using **First-Label Reduction** (only the first appearing label for each sample is taken as the target).
- **Application:** Classify data into **108 independent technique classes**, providing Top-1 to Top-3 suggested attack techniques to help SOC analysts speed up incident investigation.

## 🗂️ Dataset & Class Imbalance Handling
- **Stratified Split:** The dataset is split into Train (70%), Validation (15%), and Test (15%) sets, preserving the distribution across all 108 classes.
- **Data Augmentation:** Applied exclusively on the Train set to address class imbalance (boosting minor classes to at least 300 samples per class). Techniques used:
  - *Back Translation:* Automatically translating text to other languages and back to English.
  - *Cyber EDA:* Replacing cybersecurity synonyms using a predefined `CYBER_SYNONYMS` dictionary.

## ⚙️ Preprocessing & Tokenization
- **9-Step Pipeline:** Includes basic cleaning, normalizing Sub-techniques to Parent-techniques, rare label reduction, and preventing data leakage (anonymizing URLs and MITRE technique IDs).
- **Entity Normalization:** Identifies and tokenizes sensitive or specific entities like SQLi payloads, CVEs, IPs, URLs, Hashes, Registry Keys, and File Paths.
- **Custom Tokenization:** Uses a custom regular expression (`r"[a-z0-9_]+(?:[./:-][a-z0-9_]+)*"`) to **preserve special characters like `.` `/` `:` `-` `_`**. This ensures critical technical terms like file names (e.g., `cmd.exe`), paths (e.g., `/var/log`), and arguments (e.g., `-noprofile`) remain intact.

## 🧠 Feature Extraction & Training
- **Vectorization (Hybrid TF-IDF):** Combines two feature spaces to create a 160,000-dimensional vector:
  - *Word TF-IDF (80,000 dimensions):* Uses 1-3 word N-grams with English stopwords removed.
  - *Char TF-IDF (80,000 dimensions):* Uses 2-5 character N-grams, keeping stopwords to preserve word roots, prefixes/suffixes, and obfuscated commands.
- **Baseline Models:**
  - Logistic Regression (with `class_weight='balanced'`)
  - Linear SVC (with `class_weight='balanced'`)
  - Random Forest (with `max_depth=30`)

## 📊 Evaluation
Models are evaluated independently on the **Test set (15%)** using the following metrics:
- **Top-1 Accuracy:** Accuracy of the highest-probability prediction.
- **Top-3 Accuracy:** Accuracy when the true label is within the top 3 predictions (highly relevant for practical SOC workflows).
- **Macro F1 & Weighted F1:** Overall performance across all 108 classes.

## 🚀 Usage

Follow these steps to run the single-label classification pipeline:

1. **Preprocess the data:**
   Run the `preprocessing.py` script to clean the raw data and generate the `attack_dataset_stage1_frequent.csv` files.
   ```bash
   python src/preprocessing.py
   ```

2. **Train and evaluate models:**
   Run the two single-label classification scripts to train the models and generate predictions.
   ```bash
   SingleLabel-TF-IDF.ipynb
   SingleLabel-BM25.ipynb
   ```

3. **View the results:**
   All evaluation metrics, model logs, and outputs will be automatically saved in the `results/` directory.
