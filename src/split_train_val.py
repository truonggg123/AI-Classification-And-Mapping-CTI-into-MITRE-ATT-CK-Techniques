import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def create_fixed_train_val_split(train_file='dataset/processed/train.csv', 
                                 out_dir='dataset/processed',
                                 splits_dir='results/splits',
                                 val_fraction=0.10,
                                 seed=42):
    train_path = Path(train_file)
    out_path = Path(out_dir)
    splits_path = Path(splits_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    splits_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading official Train set from: {train_path}")
    df_train = pd.read_csv(train_path)
    
    original_train_count = len(df_train)
    
    # Parse labels for stratification
    label_lists = [str(l).split(',') for l in df_train['Labels']]
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)
    num_labels = len(mlb.classes_)
    
    train_idx_file = splits_path / 'train_indices.npy'
    val_idx_file = splits_path / 'val_indices.npy'
    meta_file = splits_path / 'split_metadata.json'
    
    reuse_split = False
    if train_idx_file.exists() and val_idx_file.exists() and meta_file.exists():
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get("original_train_count") == original_train_count and meta.get("validation_fraction") == val_fraction and meta.get("split_seed") == seed:
                print("[INFO] Existing split is compatible. Reusing split indices.")
                train_indices = np.load(train_idx_file)
                val_indices = np.load(val_idx_file)
                reuse_split = True
        except Exception as e:
            print(f"[WARNING] Could not reuse existing split: {e}")
            
    if not reuse_split:
        print(f"[INFO] Creating new fixed Train ({1-val_fraction:.0%}) / Validation ({val_fraction:.0%}) split...")
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_indices, val_indices = next(msss.split(df_train['Cleaned_Text'].values, Y))
        
        np.save(train_idx_file, train_indices)
        np.save(val_idx_file, val_indices)
        
        meta = {
            "split_seed": seed,
            "validation_fraction": val_fraction,
            "original_train_count": original_train_count,
            "clean_train_count": len(train_indices),
            "actual_train_count": len(train_indices),
            "validation_count": len(val_indices),
            "num_labels": num_labels
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4)
        print(f"[INFO] Saved split metadata and indices to: {splits_path}")

    # Create the dataframes
    df_fixed_train = df_train.iloc[train_indices].reset_index(drop=True)
    df_fixed_val = df_train.iloc[val_indices].reset_index(drop=True)
    
    # Remove text overlaps from Train to prevent leakage
    print("[INFO] Checking for duplicate texts between Train and Validation/Test...")
    val_texts = set(df_fixed_val['Cleaned_Text'].str.lower().str.strip())
    test_path = out_path / 'test.csv'
    if test_path.exists():
        df_test = pd.read_csv(test_path)
        test_texts = set(df_test['Cleaned_Text'].str.lower().str.strip())
    else:
        test_texts = set()
        
    leakage_texts = val_texts.union(test_texts)
    
    initial_train_len = len(df_fixed_train)
    # Mask for non-overlapping texts
    mask = ~df_fixed_train['Cleaned_Text'].str.lower().str.strip().isin(leakage_texts)
    df_fixed_train = df_fixed_train[mask].reset_index(drop=True)
    
    removed = initial_train_len - len(df_fixed_train)
    if removed > 0:
        print(f"[WARNING] Removed {removed} rows from Train that overlapped with Validation or Test texts.")
        
    # Save the CSV files
    train_out_path = out_path / 'train_original_fixed.csv'
    val_out_path = out_path / 'validation_original_fixed.csv'
    
    df_fixed_train.to_csv(train_out_path, index=False, encoding='utf-8')
    df_fixed_val.to_csv(val_out_path, index=False, encoding='utf-8')
    
    print(f"[SUCCESS] Fixed Train subset saved to: {train_out_path} ({len(df_fixed_train)} samples)")
    print(f"[SUCCESS] Fixed Validation subset saved to: {val_out_path} ({len(df_fixed_val)} samples)")

if __name__ == '__main__':
    create_fixed_train_val_split()
