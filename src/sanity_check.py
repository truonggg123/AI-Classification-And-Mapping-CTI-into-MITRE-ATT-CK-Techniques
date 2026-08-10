import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
from collections import Counter

def check_overlap(set1, set2, name1, name2):
    overlap = set1.intersection(set2)
    return overlap

def run_sanity_check(train_fixed_path, train_aug_path, val_fixed_path, test_path, expected_test_count=None):
    print("========================================================")
    print("DATA SPLIT & AUGMENTATION SANITY CHECK")
    print("========================================================")
    
    # Load datasets
    df_train_orig = pd.read_csv(train_fixed_path)
    df_train_aug = pd.read_csv(train_aug_path)
    df_val = pd.read_csv(val_fixed_path)
    df_test = pd.read_csv(test_path)
    
    # Normalize texts for text intersection checks
    train_texts_orig = set(df_train_orig['Cleaned_Text'].str.lower().str.strip())
    train_texts_aug = set(df_train_aug['Cleaned_Text'].str.lower().str.strip())
    val_texts = set(df_val['Cleaned_Text'].str.lower().str.strip())
    test_texts = set(df_test['Cleaned_Text'].str.lower().str.strip())
    
    all_train_texts = train_texts_orig.union(train_texts_aug)
    
    # Source sample IDs
    train_ids_orig = set(df_train_orig['source_sample_id'])
    
    # df_train_aug contains both original and synthetic. Let's separate them
    if 'is_augmented' in df_train_aug.columns:
        df_train_synth = df_train_aug[df_train_aug['is_augmented'] == 1]
    else:
        # Fallback if no flag
        df_train_synth = df_train_aug
        
    train_ids_aug = set(df_train_synth['source_sample_id'])
    val_ids = set(df_val.get('source_sample_id', []))
    test_ids = set(df_test.get('source_sample_id', []))
    
    # 1. Train source IDs ∩ Validation source IDs = 0
    overlap_train_val = check_overlap(train_ids_orig, val_ids, "Train", "Validation")
    
    # 2. Augmented Train source IDs ∩ Validation source IDs = 0
    overlap_aug_val = check_overlap(train_ids_aug, val_ids, "Augmented Train", "Validation")
    
    # 3. Train source IDs ∩ Test source IDs = 0
    overlap_train_test = check_overlap(train_ids_orig, test_ids, "Train", "Test")
    
    # 4. Normalized Train text ∩ Validation text = 0
    text_overlap_train_val = check_overlap(all_train_texts, val_texts, "Normalized Train", "Validation")
    
    # 5. Normalized Train text ∩ Test text = 0
    text_overlap_train_test = check_overlap(all_train_texts, test_texts, "Normalized Train", "Test")
    
    # 6. Validation augmented rows = 0
    val_aug_count = len(df_val[df_val['is_augmented'] == 1]) if 'is_augmented' in df_val.columns else 0
    
    # 7. Test augmented rows = 0
    test_aug_count = len(df_test[df_test['is_augmented'] == 1]) if 'is_augmented' in df_test.columns else 0
    
    # 8. Every synthetic row source_sample_id exists in original fixed Train
    synth_without_parent = train_ids_aug - train_ids_orig
    
    # 9. Every synthetic row Labels exactly matches its parent Labels
    label_mismatch_count = 0
    parent_labels = dict(zip(df_train_orig['source_sample_id'], df_train_orig['Labels']))
    for idx, row in df_train_synth.iterrows():
        sid = row['source_sample_id']
        if sid in parent_labels:
            if row['Labels'] != parent_labels[sid]:
                label_mismatch_count += 1
                
    # 10. Official Test count before/after = unchanged
    actual_test_count = len(df_test)
    test_count_issue = False
    if expected_test_count is not None and actual_test_count != expected_test_count:
        test_count_issue = True
        
    # Calculate stats for reporting
    all_labels = [lbl.strip() for sublist in df_train_orig['Labels'].dropna().str.split(',') for lbl in sublist]
    label_counts = Counter(all_labels)
    head_labels = sum(1 for c in label_counts.values() if c >= 100)
    medium_labels = sum(1 for c in label_counts.values() if 30 <= c < 100)
    tail_labels = sum(1 for c in label_counts.values() if c < 30)
    
    print(f"Official original Train: {len(df_train_orig):,}")
    print(f"Fixed Train subset: {len(df_train_orig):,}")
    print(f"Fixed Validation subset: {len(df_val):,}")
    print(f"Official Test: {len(df_test):,}")
    print(f"\nNumber of labels: {len(label_counts)}")
    print(f"Original Train samples before augmentation: {len(df_train_orig):,}")
    print(f"Synthetic generated samples: {len(df_train_synth):,}")
    print(f"Total augmented Train samples: {len(df_train_aug):,}")
    print(f"\nValidation original samples: {len(df_val):,}")
    print(f"Validation synthetic samples: {val_aug_count}")
    print(f"\nTest original samples: {len(df_test):,}")
    print(f"Test synthetic samples: {test_aug_count}")
    
    print(f"\nTrain/Validation source overlap: {len(overlap_train_val)}")
    print(f"Augmented-Train/Validation source overlap: {len(overlap_aug_val)}")
    
    print(f"\nOriginal Head labels: {head_labels}")
    print(f"Original Medium labels: {medium_labels}")
    print(f"Original Tail labels: {tail_labels}")
    
    print("========================================================")
    
    # Assertions
    failed = False
    
    if len(overlap_train_val) > 0:
        print("[ERROR] Augmentation-family leakage detected: Train and Validation share source IDs.")
        failed = True
        
    if len(overlap_aug_val) > 0:
        print("[ERROR] Augmentation-family leakage detected: Augmented Train and Validation share source IDs.")
        failed = True
        
    if len(overlap_train_test) > 0:
        print("[ERROR] Leakage detected: Train and Test share source IDs.")
        failed = True
        
    if val_aug_count > 0:
        print("[ERROR] Validation set contains augmented samples!")
        failed = True
        
    if test_aug_count > 0:
        print("[ERROR] Test set contains augmented samples!")
        failed = True
        
    if len(text_overlap_train_val) > 0:
        print(f"[ERROR] Found {len(text_overlap_train_val)} overlapping normalized texts between Train and Validation.")
        failed = True
        
    if len(text_overlap_train_test) > 0:
        print(f"[ERROR] Found {len(text_overlap_train_test)} overlapping normalized texts between Train and Test.")
        failed = True
        
    if len(synth_without_parent) > 0:
        print(f"[ERROR] Found {len(synth_without_parent)} synthetic samples with source IDs not in fixed Train.")
        failed = True
        
    if label_mismatch_count > 0:
        print(f"[ERROR] Found {label_mismatch_count} synthetic samples with labels not exactly matching their parents.")
        failed = True
        
    if test_count_issue:
        print(f"[ERROR] Test count mismatch. Expected: {expected_test_count}, Got: {actual_test_count}")
        failed = True
        
    if failed:
        print("\n[FAILED] Sanity checks failed. DO NOT START TRAINING.")
        sys.exit(1)
    else:
        print("\n[OK] Fixed Train/Validation split created or reused.")
        print("[OK] Augmentation applied to training subset only.")
        print("[OK] Validation contains original samples only.")
        print("[OK] Test contains original samples only.")
        print("[OK] No source-level overlap between Train and Validation.")
        print("[OK] Dataset is ready for leakage-safe SecureBERT training.")
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fixed', default='dataset/processed/train_original_fixed.csv')
    parser.add_argument('--train_aug', default='dataset/processed/train_augmented_eda.csv')
    parser.add_argument('--val_fixed', default='dataset/processed/validation_original_fixed.csv')
    parser.add_argument('--test', default='dataset/processed/test.csv')
    parser.add_argument('--expected_test', type=int, default=None)
    args = parser.parse_args()
    
    run_sanity_check(args.train_fixed, args.train_aug, args.val_fixed, args.test, args.expected_test)
