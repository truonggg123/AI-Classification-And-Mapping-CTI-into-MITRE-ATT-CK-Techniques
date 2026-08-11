"""
CTI & MITRE ATT&CK Dataset Merging & Deduplication Module
Loads raw datasets (Attack_Dataset.csv, single_label.json, multi_label.json),
extracts parent technique codes (Txxxx), neutralizes direct Txxxx codes to avoid label leakage,
and performs union deduplication to produce 01_merged_cti_dataset.csv.

Usage (Module):
    from src.merge_datasets import merge_raw_datasets
    df_merged = merge_raw_datasets(raw_dir='dataset/raw', output_file='dataset/processed/01_merged_cti_dataset.csv')

Usage (CLI):
    python src/merge_datasets.py --raw_dir dataset/raw --output_file dataset/processed/01_merged_cti_dataset.csv
"""

import argparse
import json
import re
from pathlib import Path
import pandas as pd

MITRE_PATTERN = re.compile(r'T\d{4}(?:\.\d{3})?')


def get_parent_label(lbl):
    """Extract parent technique ID (Txxxx) from any MITRE technique string."""
    if pd.isna(lbl):
        return None
    lbl_str = str(lbl).strip()
    match = MITRE_PATTERN.search(lbl_str)
    if match:
        return match.group(0).split('.')[0]
    return None


def clean_cti_text_preliminary(text):
    """
    Preliminary text cleaning pipeline:
    - Neutralizes direct MITRE IDs (Txxxx / Txxxx.xxx) to prevent label leakage.
    - Preserves HTML, URLs, Markdown, and entity strings for downstream tokenization/anonymization.
    - Normalizes extra whitespace while maintaining natural grammar and casing.
    """
    if pd.isna(text):
        return ""
    t = str(text)
    t = MITRE_PATTERN.sub(' ', t)  # Neutralize direct MITRE technique codes to avoid label leakage
    t = re.sub(r'\b(unknown|nan)\b', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def merge_raw_datasets(raw_dir='dataset/raw', output_file='dataset/processed/01_merged_cti_dataset.csv'):
    """
    Loads, cleans, and deduplicates raw CTI datasets:
    - dataset.csv
    - single_label.json
    - multi_label.json
    Saves and returns merged DataFrame with Cleaned_Text and Labels columns.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_csv = raw_path / 'dataset.csv'
    single_json = raw_path / 'single_label.json'
    multi_json = raw_path / 'multi_label.json'

    text_to_labels = {}
    order = []

    # 1. Process dataset.csv
    if dataset_csv.exists():
        print(f"[INFO] Parsing {dataset_csv}...")
        df_ds = pd.read_csv(dataset_csv)
        ds_count = 0
        for _, row in df_ds.iterrows():
            raw_t = str(row.get('sentence', '')).strip()
            raw_l = str(row.get('label_tec', '')).strip()
            if not raw_t or not raw_l:
                continue
            parent_l = get_parent_label(raw_l)
            if not parent_l:
                continue
            cleaned_t = clean_cti_text_preliminary(raw_t)
            if not cleaned_t:
                continue
            ds_count += 1
            if cleaned_t not in text_to_labels:
                text_to_labels[cleaned_t] = {parent_l}
                order.append(cleaned_t)
            else:
                text_to_labels[cleaned_t].add(parent_l)
        print(f"   [RESULT] Processed {ds_count:,} valid entries from dataset.csv")

    # 2. Process single_label.json
    if single_json.exists():
        print(f"[INFO] Parsing {single_json}...")
        with open(single_json, 'r', encoding='utf-8') as f:
            single_data = json.load(f)
        s_count = 0
        for item in single_data:
            raw_t = str(item.get('text', '')).strip()
            raw_l = str(item.get('label', '')).strip()
            if not raw_t or not raw_l:
                continue
            parent_l = get_parent_label(raw_l)
            if not parent_l:
                continue
            cleaned_t = clean_cti_text_preliminary(raw_t)
            if not cleaned_t:
                continue
            s_count += 1
            if cleaned_t not in text_to_labels:
                text_to_labels[cleaned_t] = {parent_l}
                order.append(cleaned_t)
            else:
                text_to_labels[cleaned_t].add(parent_l)
        print(f"   [RESULT] Processed {s_count:,} valid entries from single_label.json")

    # 3. Process multi_label.json
    if multi_json.exists():
        print(f"[INFO] Parsing {multi_json}...")
        with open(multi_json, 'r', encoding='utf-8') as f:
            multi_data = json.load(f)
        m_count = 0
        for item in multi_data:
            labels_list = item.get('labels', [])
            if not labels_list:
                continue
            raw_t = str(item.get('sentence', '')).strip()
            if not raw_t:
                continue
            parent_labels = set(get_parent_label(str(l)) for l in labels_list if get_parent_label(str(l)))
            if not parent_labels:
                continue
            cleaned_t = clean_cti_text_preliminary(raw_t)
            if not cleaned_t:
                continue
            m_count += 1
            if cleaned_t not in text_to_labels:
                text_to_labels[cleaned_t] = parent_labels
                order.append(cleaned_t)
            else:
                text_to_labels[cleaned_t].update(parent_labels)
        print(f"   [RESULT] Processed {m_count:,} valid entries from multi_label.json")

    final_rows = []
    for text in order:
        sorted_labels = ','.join(sorted(list(text_to_labels[text])))
        final_rows.append({'Cleaned_Text': text, 'Labels': sorted_labels})

    df_merged = pd.DataFrame(final_rows)
    df_merged.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[SUCCESS] Exported merged dataset to: {output_path} ({len(df_merged):,} unique samples)")
    return df_merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CTI ATT&CK Raw Dataset Merging & Deduplication Script")
    parser.add_argument('--raw_dir', type=str, default='dataset/raw', help='Path to raw datasets directory')
    parser.add_argument('--output_file', type=str, default='dataset/processed/01_merged_cti_dataset.csv', help='Path to merged CSV output file')

    args = parser.parse_args()
    merge_raw_datasets(raw_dir=args.raw_dir, output_file=args.output_file)
