"""
CTI & MITRE ATT&CK Preprocessing, Anonymization & Stratification Module
Loads merged CTI datasets, filters out high multi-label outliers (<= 3 labels),
anonymizes variable technical entities ([CVE], [IPV4], [URL], [FILE_PATH], [HASH]),
tokenizes CTI text for TF-IDF baselines, binarizes dynamic target labels, and performs
a 70/10/20 multi-label stratified train/val/test split.

Usage (Module):
    from src.preprocessing import run_preprocessing_pipeline
    run_preprocessing_pipeline(target_dataset='all')

Usage (CLI):
    python src/preprocessing.py --target_dataset all
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


def run_preprocessing_pipeline(input_file=None, processed_dir=None, results_dir='results', target_dataset='joint', max_labels=3, train_size=0.70, val_size=0.10, test_size=0.20, random_state=42):
    """
    Loads merged CTI dataset, filters outliers (<= max_labels), anonymizes entities,
    tokenizes text, binarizes targets, performs 70/10/20 stratified train/val/test split, and exports artifacts.
    """
    # Resolve input_file and processed_dir if not explicitly set
    if processed_dir is None:
        processed_path = Path('dataset/processed') / target_dataset
    else:
        processed_path = Path(processed_dir)

    if input_file is None or not Path(input_file).exists():
        possible_inputs = [
            processed_path / 'raw_merged.csv',
            Path('dataset/processed') / target_dataset / 'raw_merged.csv',
            Path('dataset/processed/joint/raw_merged.csv'),
            Path('dataset/processed/cti_to_mitre/raw_merged.csv'),
            Path('dataset/processed/tram/raw_merged.csv'),
            Path('dataset/processed/01_merged_cti_dataset.csv')
        ]
        found_input = None
        for p in possible_inputs:
            if p.exists():
                found_input = p
                break
        if found_input is None:
            raise FileNotFoundError(
                f"Cannot find merged dataset file. Please run `python src/merge_datasets.py --target_dataset {target_dataset}` first."
            )
        input_path = found_input
    else:
        input_path = Path(input_file)
        
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
    
    print("\n=== STEP 3: ENTITY ANONYMIZATION, DEDUPLICATION & DOMAIN TOKENIZATION ===")
    df_filtered['Cleaned_Text'] = df_filtered['Cleaned_Text'].apply(anonymize_cti_text)
    df_filtered = df_filtered[df_filtered['Cleaned_Text'].str.len() > 0].copy()
    
    # Strict normalized deduplication to prevent Train/Val/Test data leakage
    def norm_key(s):
        return re.sub(r'\s+', ' ', str(s).lower()).strip()
    
    df_filtered['norm_key'] = df_filtered['Cleaned_Text'].apply(norm_key)
    
    def union_labels_func(series):
        all_lbls = set()
        for x in series:
            for l in str(x).split(','):
                l_clean = l.strip()
                if l_clean and l_clean != 'nan':
                    all_lbls.add(l_clean)
        return ','.join(sorted(all_lbls))
    
    df_dedup = df_filtered.groupby('norm_key', as_index=False).agg({
        'Cleaned_Text': 'first',
        'Labels': union_labels_func
    })
    df_dedup['Tokenized_Text'] = df_dedup['Cleaned_Text'].apply(tokenize_cti_text)
    df_dedup['Label_Count'] = df_dedup['Labels'].apply(lambda x: len([l for l in str(x).split(',') if l.strip()]))
    df_processed = df_dedup[df_dedup['Label_Count'] <= max_labels].copy().reset_index(drop=True)
    df_processed['source_sample_id'] = df_processed.index + 1
    
    if len(df_filtered) != len(df_processed):
        print(f"[INFO] Merged {len(df_filtered) - len(df_processed)} duplicate instances into unified unique text samples.")
    
    processed_output_path = processed_path / '02_processed_cti_dataset.csv'
    df_processed[['Cleaned_Text', 'Labels', 'Label_Count', 'Tokenized_Text', 'source_sample_id']].to_csv(processed_output_path, index=False, encoding='utf-8')
    print(f"[INFO] Saved preprocessed dataset to: {processed_output_path}")
    
    print("\n=== STEP 4: LABEL BINARIZATION (DYNAMIC TARGET SPACE) ===")
    label_lists = [str(l).split(',') for l in df_processed['Labels']]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)
    
    binarizer_path = processed_path / 'multilabel_binarizer.pkl'
    with open(binarizer_path, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"[INFO] Saved MultiLabelBinarizer to: {binarizer_path} ({len(mlb.classes_)} classes)")
    
    print("\n=== STEP 5: MULTI-LABEL STRATIFIED TRAIN/VAL/TEST SPLIT (70/10/20) ===")
    # Step 5a: Split off test set (20%)
    msss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_indices, test_indices = next(msss_test.split(df_processed['Cleaned_Text'].values, Y))
    
    # Step 5b: Split remaining 80% into Train (70%) and Val (10%)
    val_ratio_of_train_val = val_size / (train_size + val_size)
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio_of_train_val, random_state=random_state)
    train_sub_indices, val_sub_indices = next(msss_val.split(
        df_processed['Cleaned_Text'].iloc[train_val_indices].values,
        Y[train_val_indices]
    ))
    
    train_indices = train_val_indices[train_sub_indices]
    val_indices = train_val_indices[val_sub_indices]
    
    df_train = df_processed.iloc[train_indices].reset_index(drop=True)
    df_val = df_processed.iloc[val_indices].reset_index(drop=True)
    df_test = df_processed.iloc[test_indices].reset_index(drop=True)
    
    train_csv_path = processed_path / 'train.csv'
    val_csv_path = processed_path / 'val.csv'
    test_csv_path = processed_path / 'test.csv'
    
    df_train.to_csv(train_csv_path, index=False, encoding='utf-8')
    df_val.to_csv(val_csv_path, index=False, encoding='utf-8')
    df_test.to_csv(test_csv_path, index=False, encoding='utf-8')
    
    train_label_coverage = (Y[train_indices].sum(axis=0) > 0).sum()
    val_label_coverage = (Y[val_indices].sum(axis=0) > 0).sum()
    test_label_coverage = (Y[test_indices].sum(axis=0) > 0).sum()
    
    print(f"[INFO] Train samples: {len(df_train):,} ({len(df_train)/len(df_processed)*100:.2f}%) | Label coverage: {train_label_coverage}/{len(mlb.classes_)}")
    print(f"[INFO] Val samples  : {len(df_val):,} ({len(df_val)/len(df_processed)*100:.2f}%) | Label coverage: {val_label_coverage}/{len(mlb.classes_)}")
    print(f"[INFO] Test samples : {len(df_test):,} ({len(df_test)/len(df_processed)*100:.2f}%) | Label coverage: {test_label_coverage}/{len(mlb.classes_)}")
    
    print("\n=== STEP 6: EXPORT PREPROCESSING REPORT ===")
    report_data = {
        "target_dataset": target_dataset,
        "initial_samples": initial_sample_count,
        "removed_outliers_count": removed_outlier_count,
        "outlier_filter_rule": f"Label_Count <= {max_labels}",
        "final_valid_samples": len(df_processed),
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "train_ratio": round(len(df_train) / len(df_processed), 4),
        "val_ratio": round(len(df_val) / len(df_processed), 4),
        "test_ratio": round(len(df_test) / len(df_processed), 4),
        "total_unique_target_labels": len(mlb.classes_),
        "train_label_coverage": int(train_label_coverage),
        "val_label_coverage": int(val_label_coverage),
        "test_label_coverage": int(test_label_coverage),
        "random_seed": random_state,
        "split_method": "MultilabelStratifiedShuffleSplit (70/10/20)",
        "anonymized_entities": ["CVE", "IPV4", "URL", "FILE_PATH", "HASH"],
        "output_files": {
            "processed_dataset": str(processed_output_path),
            "train_set": str(train_csv_path),
            "val_set": str(val_csv_path),
            "test_set": str(test_csv_path),
            "multilabel_binarizer": str(binarizer_path)
        }
    }
    
    report_json_path = results_path / f'02_preprocessing_report_{target_dataset}.json'
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
        
    print(f"[SUCCESS] Preprocessing report saved to: {report_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CTI ATT&CK Preprocessing, Anonymization & Stratification Script")
    parser.add_argument('--input_file', type=str, default=None, help='Path to merged input CSV file (ignored when --target_dataset=all)')
    parser.add_argument('--target_dataset', type=str, default='all', choices=['cti_to_mitre', 'tram', 'joint', 'all'], help='Target dataset to preprocess. Use "all" to preprocess all 3 datasets at once.')
    parser.add_argument('--processed_dir', type=str, default=None, help='Path to processed output directory (ignored when --target_dataset=all)')
    parser.add_argument('--results_dir', type=str, default='results', help='Path to results directory')
    parser.add_argument('--max_labels', type=int, default=3, help='Maximum label count threshold per sample')
    parser.add_argument('--train_size', type=float, default=0.70, help='Train set split ratio (default: 0.70)')
    parser.add_argument('--val_size', type=float, default=0.10, help='Validation set split ratio (default: 0.10)')
    parser.add_argument('--test_size', type=float, default=0.20, help='Test set split ratio (default: 0.20)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for stratification')
    
    args = parser.parse_args()
    if args.target_dataset == 'all':
        for ds in ['cti_to_mitre', 'tram', 'joint']:
            print(f"\n{'='*60}")
            print(f"[PREPROCESSING] target_dataset = {ds}")
            print(f"{'='*60}")
            run_preprocessing_pipeline(
                input_file=None,
                target_dataset=ds,
                processed_dir=None,
                results_dir=args.results_dir,
                max_labels=args.max_labels,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                random_state=args.seed
            )
        print("\n[DONE] All 3 datasets preprocessed successfully.")
    else:
        run_preprocessing_pipeline(
            input_file=args.input_file,
            target_dataset=args.target_dataset,
            processed_dir=args.processed_dir,
            results_dir=args.results_dir,
            max_labels=args.max_labels,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.seed
        )
