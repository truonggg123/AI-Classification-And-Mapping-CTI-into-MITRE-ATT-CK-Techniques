"""
CTI & MITRE ATT&CK Preprocessing, Anonymization & Stratification Module
Loads 01_merged_cti_dataset.csv, filters out high multi-label outliers (<= 3 labels),
anonymizes variable technical entities ([CVE], [IPV4], [URL], [FILE_PATH], [HASH]),
tokenizes CTI text for TF-IDF baselines, binarizes 378 target labels, and performs
an 80/20 multi-label stratified train/test split.

Usage (Module):
    from src.preprocessing import run_preprocessing_pipeline
    run_preprocessing_pipeline(input_file='dataset/processed/01_merged_cti_dataset.csv', processed_dir='dataset/processed', results_dir='results')

Usage (CLI):
    python src/preprocessing.py --input_file dataset/processed/01_merged_cti_dataset.csv --processed_dir dataset/processed --results_dir results
"""

import argparse
import json
import re
import html
import unicodedata
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Regex Patterns for Variable Technical Entities
REG_CVE = re.compile(r'(?i)CVE-\d{4}-\d{4,7}')
REG_IPV4 = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
REG_URL = re.compile(r'(?i)https?://[^\s]+|www\.[^\s]+')
REG_WIN_PATH = re.compile(r'[A-Za-z]:\\[^\s]+')
REG_UNIX_PATH = re.compile(r'/(?:[a-zA-Z0-9_\.-]+/)+[a-zA-Z0-9_\.-]+')
REG_HASH = re.compile(r'\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b')
REG_HTML = re.compile(r'<[^>]+>')
REG_MARKDOWN = re.compile(r'[\*\_`#]')
CTI_TOKEN_PATTERN = r"[a-z0-9_\[\]]+(?:[./:-][a-z0-9_\[\]]+)*"

TOTAL_MITRE_PARENT_TECHNIQUES = 378


def anonymize_cti_text(text):
    """
    Standalone entity anonymization pipeline:
    - Unescapes HTML entities & applies NFKC Unicode normalization.
    - Cleans HTML tags and Markdown formatting.
    - Anonymizes CVE, IPV4, URL, FILE_PATH, and HASH entities with special tokens.
    - Preserves natural sentence structure and casing for Transformer models.
    """
    if pd.isna(text):
        return ""
    t = str(text)
    t = html.unescape(t)
    t = unicodedata.normalize('NFKC', t)
    t = REG_HTML.sub(' ', t)
    t = REG_MARKDOWN.sub(' ', t)
    
    # Anonymize technical entities
    t = REG_CVE.sub(' [CVE] ', t)
    t = REG_URL.sub(' [URL] ', t)
    t = REG_WIN_PATH.sub(' [FILE_PATH] ', t)
    t = REG_UNIX_PATH.sub(' [FILE_PATH] ', t)
    t = REG_IPV4.sub(' [IPV4] ', t)
    t = REG_HASH.sub(' [HASH] ', t)
    
    # Clean step noise and whitespace
    t = re.sub(r'\b(step|phase)\s+\d+\b', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(unknown|nan)\b', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def cti_tokenizer(text):
    """Extract lowercased CTI technical tokens preserving special entity placeholders."""
    return re.findall(CTI_TOKEN_PATTERN, str(text).lower())


def tokenize_cti_text(text):
    """Convert text into space-delimited lowercased token string."""
    tokens = cti_tokenizer(text)
    return " ".join(tokens)


def run_preprocessing_pipeline(input_file='dataset/processed/01_merged_cti_dataset.csv', processed_dir='dataset/processed', results_dir='results', max_labels=3, test_size=0.20, random_state=42):
    """
    Loads merged CTI dataset, filters outliers (<= max_labels), anonymizes entities,
    tokenizes text, binarizes targets, performs stratified split, and exports artifacts.
    """
    input_path = Path(input_file)
    processed_path = Path(processed_dir)
    results_path = Path(results_dir)
    
    processed_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=== STEP 1: LOAD MERGED DATASET ({input_path}) ===")
    df_raw = pd.read_csv(input_path)
    initial_sample_count = len(df_raw)
    print(f"[INFO] Loaded merged dataset: {initial_sample_count:,} samples")
    
    print(f"\n=== STEP 2: OUTLIER FILTERING (Label_Count <= {max_labels}) ===")
    label_lists_raw = [str(l).split(',') for l in df_raw['Labels']]
    df_raw['Label_Count'] = [len(l) for l in label_lists_raw]
    
    df_filtered = df_raw[df_raw['Label_Count'] <= max_labels].copy().reset_index(drop=True)
    removed_outlier_count = initial_sample_count - len(df_filtered)
    print(f"[INFO] Removed {removed_outlier_count} outlier samples with > {max_labels} labels.")
    print(f"[INFO] Retained {len(df_filtered):,} valid samples.")
    
    print("\n=== STEP 3: ENTITY ANONYMIZATION & DOMAIN TOKENIZATION ===")
    df_filtered['Cleaned_Text'] = df_filtered['Cleaned_Text'].apply(anonymize_cti_text)
    df_filtered['Tokenized_Text'] = df_filtered['Cleaned_Text'].apply(tokenize_cti_text)
    df_processed = df_filtered[df_filtered['Cleaned_Text'].str.len() > 0].reset_index(drop=True)
    
    processed_output_path = processed_path / '02_processed_cti_dataset.csv'
    df_processed.to_csv(processed_output_path, index=False, encoding='utf-8')
    print(f"[INFO] Saved preprocessed dataset to: {processed_output_path}")
    
    print("\n=== STEP 4: LABEL BINARIZATION (378 TARGET SPACE) ===")
    label_lists = [str(l).split(',') for l in df_processed['Labels']]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)
    
    binarizer_path = processed_path / 'multilabel_binarizer.pkl'
    with open(binarizer_path, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"[INFO] Saved MultiLabelBinarizer to: {binarizer_path} ({len(mlb.classes_)} classes)")
    
    print("\n=== STEP 5: MULTI-LABEL STRATIFIED TRAIN/TEST SPLIT ===")
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(msss.split(df_processed['Cleaned_Text'].values, Y))
    
    df_train = df_processed.iloc[train_indices].reset_index(drop=True)
    df_test = df_processed.iloc[test_indices].reset_index(drop=True)
    
    train_csv_path = processed_path / 'train.csv'
    test_csv_path = processed_path / 'test.csv'
    df_train.to_csv(train_csv_path, index=False, encoding='utf-8')
    df_test.to_csv(test_csv_path, index=False, encoding='utf-8')
    
    train_label_coverage = (Y[train_indices].sum(axis=0) > 0).sum()
    test_label_coverage = (Y[test_indices].sum(axis=0) > 0).sum()
    
    print(f"[INFO] Train samples: {len(df_train):,} ({len(df_train)/len(df_processed)*100:.2f}%) | Label coverage: {train_label_coverage}/{len(mlb.classes_)}")
    print(f"[INFO] Test samples : {len(df_test):,} ({len(df_test)/len(df_processed)*100:.2f}%) | Label coverage: {test_label_coverage}/{len(mlb.classes_)}")
    
    print("\n=== STEP 6: EXPORT PREPROCESSING REPORT ===")
    report_data = {
        "initial_samples": initial_sample_count,
        "removed_outliers_count": removed_outlier_count,
        "outlier_filter_rule": f"Label_Count <= {max_labels}",
        "final_valid_samples": len(df_processed),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "train_ratio": round(len(df_train) / len(df_processed), 4),
        "test_ratio": round(len(df_test) / len(df_processed), 4),
        "total_unique_target_labels": len(mlb.classes_),
        "train_label_coverage": int(train_label_coverage),
        "test_label_coverage": int(test_label_coverage),
        "random_seed": random_state,
        "split_method": "MultilabelStratifiedShuffleSplit",
        "anonymized_entities": ["CVE", "IPV4", "URL", "FILE_PATH", "HASH"],
        "output_files": {
            "processed_dataset": "dataset/processed/02_processed_cti_dataset.csv",
            "train_set": "dataset/processed/train.csv",
            "test_set": "dataset/processed/test.csv",
            "multilabel_binarizer": "dataset/processed/multilabel_binarizer.pkl"
        }
    }
    
    report_json_path = results_path / '02_preprocessing_report.json'
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
        
    print(f"[SUCCESS] Preprocessing report saved to: {report_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CTI ATT&CK Preprocessing, Anonymization & Stratification Script")
    parser.add_argument('--input_file', type=str, default='dataset/processed/01_merged_cti_dataset.csv', help='Path to merged input CSV file')
    parser.add_argument('--processed_dir', type=str, default='dataset/processed', help='Path to processed output directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Path to results directory')
    parser.add_argument('--max_labels', type=int, default=3, help='Maximum label count threshold per sample')
    parser.add_argument('--test_size', type=float, default=0.20, help='Test set split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for stratification')
    
    args = parser.parse_args()
    run_preprocessing_pipeline(
        input_file=args.input_file,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        max_labels=args.max_labels,
        test_size=args.test_size,
        random_state=args.seed
    )
