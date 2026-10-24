"""
Script: extract_label_preservation_samples.py
Description: Randomly samples 100 augmented sentences from CyberEDA and GenericEDA 
             training sets (joint dataset) to conduct human validation of the 
             label-preservation assumption: y(x) == y(Augment(x)).
Author: Research Team
Date: 2026-10-10
"""

import pandas as pd
from pathlib import Path

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

DATASET = 'joint'
N_SAMPLES = 100
SEED = 42

CYBER_PATH = BASE_DIR / f'dataset/processed/{DATASET}/train_augmented_eda.csv'
GENERIC_PATH = BASE_DIR / f'dataset/processed/{DATASET}/train_augmented_generic_eda.csv'

def extract_and_format(file_path: Path, method_name: str, annotator_pair: tuple, out_file_name: str):
    """
    Extracts N_SAMPLES augmented instances, maps them to their source text,
    computes word-level diffs, and formats output dataframe for human evaluation.
    """
    print(f"Loading {method_name} from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Map original text from rows where is_augmented == 0
    orig_rows = df[df['is_augmented'] == 0]
    orig_dict = orig_rows.set_index('source_sample_id')['Cleaned_Text'].to_dict()
    
    # Filter augmented candidates (is_augmented == 1)
    aug_rows = df[df['is_augmented'] == 1].copy()
    
    # Reproducible random sampling
    sampled = aug_rows.sample(n=min(N_SAMPLES, len(aug_rows)), random_state=SEED).reset_index(drop=True)
    
    a_first, a_second = annotator_pair
    records = []
    for idx, row in sampled.iterrows():
        src_id = row['source_sample_id']
        orig_text = str(orig_dict.get(src_id, ''))
        aug_text = str(row['Cleaned_Text'])
        labels = str(row['Labels'])
        
        # Word-level difference analysis
        orig_words = orig_text.split()
        aug_words = aug_text.split()
        
        # Tokens deleted (Random Deletion) and tokens inserted/substituted
        removed = [w for w in orig_words if w not in aug_words]
        added = [w for w in aug_words if w not in orig_words]
        
        records.append({
            'Sample_ID': idx + 1,
            'Source_ID': src_id,
            'Labels': labels,
            'Original_Text': orig_text,
            'Augmented_Text': aug_text,
            'Method': method_name,
            'Words_Removed_RD': ", ".join(removed) if removed else "(None)",
            'Words_Added_Replaced': ", ".join(added) if added else "(None)",
            # First annotator columns
            f'Label_Preserved_{a_first} (Y/N)': '',
            f'Semantic_Preserved_{a_first} (Y/P/N)': '',
            f'Grammar_OK_{a_first} (Y/N)': '',
            # Second annotator columns
            f'Label_Preserved_{a_second} (Y/N)': '',
            f'Semantic_Preserved_{a_second} (Y/P/N)': '',
            f'Grammar_OK_{a_second} (Y/N)': '',
            # Lead annotator / Arbiter consensus columns
            'Final_Consensus_Label (Y/N)': '',
            'Notes_Arbiter': ''
        })
        
    df_out = pd.DataFrame(records)
    out_path = SCRIPT_DIR / out_file_name
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[DONE] Successfully exported {len(df_out)} samples to: {out_path}")

if __name__ == '__main__':
    # CyberEDA evaluated by Annotator 1 and Annotator 2
    extract_and_format(
        CYBER_PATH, 
        'CyberEDA', 
        ('Annotator_1', 'Annotator_2'), 
        'eval_label_preservation_cyber.csv'
    )
    
    # GenericEDA evaluated by Annotator 3 and Annotator 4
    extract_and_format(
        GENERIC_PATH, 
        'GenericEDA', 
        ('Annotator_3', 'Annotator_4'), 
        'eval_label_preservation_generic.csv'
    )
    
    print("\n[COMPLETE] Both evaluation sheets generated successfully in analysis/label_preservation/")
