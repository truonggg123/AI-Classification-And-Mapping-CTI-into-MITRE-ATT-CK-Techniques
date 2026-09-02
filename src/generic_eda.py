"""
Generic WordNet EDA Data Augmentation Module for CTI & MITRE ATT&CK
===================================================================
Baseline Control Group for NCKH Experiments (Scenario G0 & G1).
Uses standard WordNet (NLTK) without STIX Domain-Specific Knowledge Base,
preserving exact Multi-Label Greedy Sampling parity, Seed 42, and Empirical Mean Target Count.

Usage:
    python src/generic_eda.py --target_dataset all
    python src/generic_eda.py --target_dataset joint
    python src/generic_eda.py --train_file dataset/processed/cti_to_mitre/train.csv
"""

import os
import sys
import re
import math
import random
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure NLTK and WordNet are available
try:
    import nltk
    from nltk.corpus import wordnet
    try:
        wordnet.synsets("test")
    except (LookupError, AttributeError):
        print("[INFO] Downloading NLTK wordnet & omw-1.4...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
except ImportError:
    raise ImportError("NLTK is required for Generic WordNet EDA. Please install via: pip install nltk")

SPECIAL_TOKENS = ["[CVE]", "[URL]", "[FILE_PATH]", "[IPV4]", "[HASH]"]

def is_special_token(word):
    """Detects technical entity placeholders to preserve tokenizer integrity."""
    clean = word.strip().upper()
    return clean in SPECIAL_TOKENS or (clean.startswith('[') and clean.endswith(']'))

def is_protected(word, protected_set=None):
    """Preserves technical entity tokens, identical to Cyber EDA."""
    return is_special_token(word)

def get_wordnet_synonyms(word):
    """Extracts general English synonyms for a word using WordNet."""
    clean_word = word.lower().strip(".,;:!?()\"'")
    if len(clean_word) <= 2 or is_special_token(word):
        return []
    
    synonyms = set()
    for syn in wordnet.synsets(clean_word):
        for lemma in syn.lemmas():
            syn_name = lemma.name().replace('_', ' ').strip()
            if syn_name.lower() != clean_word and len(syn_name) > 0:
                synonyms.add(syn_name)
    return list(synonyms)


# ===========================================================================
# Generic WordNet EDA Operations (Aligned with Cyber EDA Pipeline)
# ===========================================================================

def generic_synonym_replacement(words, n):
    """Replaces up to n words with standard WordNet synonyms, preserving capitalization & punctuation."""
    if len(words) <= 1:
        return words
    
    new_words = words.copy()
    candidate_indices = [
        i for i, w in enumerate(new_words) 
        if not is_protected(w) and len(get_wordnet_synonyms(w)) > 0
    ]
    
    if not candidate_indices:
        return new_words
    
    random.shuffle(candidate_indices)
    num_replaced = 0
    for idx in candidate_indices:
        if num_replaced >= n:
            break
        orig_word = new_words[idx]
        synonyms = get_wordnet_synonyms(orig_word)
        if synonyms:
            synonym = random.choice(synonyms)
            syn_words = synonym.split()
            
            # Preserve capitalization of first word
            if orig_word and orig_word[0].isupper():
                syn_words[0] = syn_words[0].capitalize()
                
            # Preserve trailing punctuation from last replaced word if present
            punct = ""
            for char in reversed(orig_word):
                if char in ".,;:!?":
                    punct = char + punct
                else:
                    break
            if punct:
                syn_words[-1] = syn_words[-1] + punct
                
            new_words[idx:idx+1] = syn_words
            num_replaced += 1
            
    return new_words


def add_wordnet_synonym(new_words):
    """Finds a candidate word in sentence, fetches WordNet synonym, and inserts at random position."""
    if not new_words:
        return
    candidate_words = [w for w in new_words if not is_protected(w)]
    if not candidate_words:
        return
    random_word = random.choice(candidate_words)
    synonyms = get_wordnet_synonyms(random_word)
    if synonyms:
        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(new_words))
        syn_words = random_synonym.split()
        for offset, w in enumerate(syn_words):
            new_words.insert(random_idx + offset, w)


def generic_random_insertion(words, n):
    """Finds WordNet synonyms for words in sentence and inserts them at random positions n times."""
    if len(words) <= 1:
        return words
    
    new_words = words.copy()
    for _ in range(n):
        add_wordnet_synonym(new_words)
    return new_words


def swap_word(new_words, protected_set=None):
    """Swaps two non-protected words, preserving punctuation at the original positions."""
    if len(new_words) <= 1:
        return new_words
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 50:
            return new_words
    
    if is_protected(new_words[random_idx_1], protected_set) or is_protected(new_words[random_idx_2], protected_set):
        return new_words
        
    # Preserve trailing punctuation at the original position in sentence
    def split_punct(w):
        clean = w.rstrip(".,;:!?")
        punct = w[len(clean):]
        return clean, punct

    w1_clean, w1_punct = split_punct(new_words[random_idx_1])
    w2_clean, w2_punct = split_punct(new_words[random_idx_2])

    new_words[random_idx_1] = w2_clean + w1_punct
    new_words[random_idx_2] = w1_clean + w2_punct
    return new_words


def generic_random_swap(words, n, protected_set=None):
    """Randomly swaps two words in the sentence n times, preserving entity tokens and punctuation."""
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words, protected_set)
    return new_words


def generic_random_deletion(words, p, protected_set=None, min_words=5):
    """
    Randomly deletes words with probability p, keeping protected entity tokens.
    Guarantees the output retains at least min_words (or original length if shorter)
    to prevent semantic collapse into 1-2 word fragments.
    """
    if len(words) <= 1:
        return words
    
    target_min = min(len(words), min_words)
    
    new_words = []
    for word in words:
        if is_protected(word, protected_set):
            new_words.append(word)
            continue
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
            
    # Guard against over-deletion: preserve at least target_min words
    if len(new_words) < target_min:
        chosen_indices = sorted(random.sample(range(len(words)), target_min))
        protected_indices = {i for i, w in enumerate(words) if is_protected(w, protected_set)}
        all_chosen = sorted(set(chosen_indices) | protected_indices)
        new_words = [words[i] for i in all_chosen]
        
    return new_words


def apply_generic_eda(text, alpha_sr=0.13, alpha_ri=0.13, alpha_rs=0.13, p_rd=0.13, protected_set=None):
    """
    Generic WordNet EDA sequential pipeline (SR -> RI -> RS -> RD):
    Follows the exact sequential pipeline structure of Cyber EDA, differing ONLY in synonym source (WordNet vs STIX).
    - alpha_sr: Synonym Replacement ratio
    - alpha_ri: Random Insertion ratio
    - alpha_rs: Random Swap ratio
    - p_rd: Random Deletion probability
    """
    if not text or not str(text).strip():
        return text
    
    words = str(text).split()
    num_words = len(words)
    if num_words == 0:
        return text
    
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))
    
    words = generic_synonym_replacement(words, n_sr)
    words = generic_random_insertion(words, n_ri)
    words = generic_random_swap(words, n_rs, protected_set)
    words = generic_random_deletion(words, p_rd, protected_set)
        
    return " ".join(words)


def single_generic_eda(text, op, alpha_sr=0.13, alpha_ri=0.13, alpha_rs=0.13, p_rd=0.13, protected_set=None):
    """
    Applies a single atomic Generic EDA operation (SR, RI, RS, or RD).
    """
    words = str(text).split()
    num_words = len(words)
    if num_words == 0:
        return text

    if op in ['sr', 'synonym']:
        n_sr = max(1, int(alpha_sr * num_words))
        words = generic_synonym_replacement(words, n_sr)
    elif op in ['ri', 'insert']:
        n_ri = max(1, int(alpha_ri * num_words))
        words = generic_random_insertion(words, n_ri)
    elif op in ['rs', 'swap']:
        n_rs = max(1, int(alpha_rs * num_words))
        words = generic_random_swap(words, n_rs, protected_set)
    elif op in ['rd', 'delete']:
        words = generic_random_deletion(words, p_rd, protected_set)
    else:
        raise ValueError(f"Unknown single EDA operation: {op}")

    return " ".join(words)


# ===========================================================================
# Pipeline Execution & Multi-Label Greedy Resampling
# ===========================================================================

def run_generic_eda_for_benchmark(
    train_file='dataset/processed/joint/train.csv',
    target_count=None,
    alpha_sr=0.13,
    alpha_ri=0.13,
    alpha_rs=0.13,
    p_rd=0.13,
    seed=42,
    save_csv=True,
    output_filename="train_augmented_generic_eda.csv"
):
    """
    Executes Generic WordNet EDA on a benchmark training set.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Input training set not found: {train_path}")
        
    print("=" * 80)
    print(f"[START] GENERIC WORDNET EDA PIPELINE: {train_path}")
    print(f"Ratios: SR={alpha_sr}, RI={alpha_ri}, RS={alpha_rs}, RD={p_rd} | Seed={seed}")
    print("=" * 80)
    
    df_train = pd.read_csv(train_path)
    df_train['Cleaned_Text'] = df_train['Cleaned_Text'].fillna("").astype(str)
    
    # Parse labels
    df_train['Label_List'] = df_train['Labels'].apply(lambda x: str(x).split(','))
    
    dynamic_label_counts = Counter([lbl for sublist in df_train['Label_List'] for lbl in sublist])
    print(f"[INFO] Total distinct MITRE classes in train set: {len(dynamic_label_counts)}")
    
    # Auto-resolve Target Count to Empirical MEAN
    if target_count is None or target_count <= 0:
        target_count = int(round(np.mean(list(dynamic_label_counts.values()))))
        print(f"[INFO] Auto-resolved target_count to empirical dataset MEAN: {target_count} samples/class")
    else:
        print(f"[INFO] Using explicitly configured target_count: {target_count} samples/class")
        
    if 'is_augmented' not in df_train.columns:
        df_train['is_augmented'] = 0
        
    print(f"[INFO] Initial Train Set Size: {len(df_train):,} samples.")
    
    # Anti-leakage filter against val.csv and test.csv
    forbidden_eval_texts = set()
    for eval_file in ['val.csv', 'test.csv']:
        eval_path = train_path.parent / eval_file
        if eval_path.exists():
            df_eval = pd.read_csv(eval_path)
            if 'Cleaned_Text' in df_eval.columns:
                for t in df_eval['Cleaned_Text'].dropna():
                    norm_t = re.sub(r'\s+', ' ', str(t).lower()).strip()
                    if norm_t:
                        forbidden_eval_texts.add(norm_t)
    if forbidden_eval_texts:
        print(f"[INFO] Loaded {len(forbidden_eval_texts):,} validation/test texts into strict anti-leakage filter.")
        
    minority_classes = {lbl: dynamic_label_counts[lbl] for lbl in dynamic_label_counts if dynamic_label_counts[lbl] < target_count}
    print(f"[INFO] Classes requiring augmentation (< {target_count} samples): {len(minority_classes)}")
    
    # Label to indices lookup
    label_to_indices = {}
    for idx, row in df_train.iterrows():
        for lbl in row['Label_List']:
            if lbl not in label_to_indices:
                label_to_indices[lbl] = []
            label_to_indices[lbl].append(idx)
            
    # Multi-Label Aware Greedy Selection
    print("[INFO] Performing Multi-Label Aware Greedy Resampling...")
    indices_to_augment = []
    sorted_minority_labels = sorted(minority_classes.keys(), key=lambda l: dynamic_label_counts[l])
    
    for lbl in sorted_minority_labels:
        available_indices = label_to_indices[lbl]
        while dynamic_label_counts[lbl] < target_count:
            sampled_idx = random.choice(available_indices)
            indices_to_augment.append(sampled_idx)
            for co_lbl in df_train.iloc[sampled_idx]['Label_List']:
                dynamic_label_counts[co_lbl] += 1
                
    print(f"[INFO] Total samples selected for Generic EDA augmentation: {len(indices_to_augment):,}")
    
    # Generate augmented rows
    augmented_records = []
    for loop_i, idx in enumerate(tqdm(indices_to_augment, desc="Generating Generic WordNet EDA Samples")):
        original_row = df_train.iloc[idx]
        original_text = original_row['Cleaned_Text']
        
        augmented_text = original_text
        for attempt in range(5):
            candidate_text = apply_generic_eda(
                original_text,
                alpha_sr=alpha_sr,
                alpha_ri=alpha_ri,
                alpha_rs=alpha_rs,
                p_rd=p_rd
            )
            candidate_text = re.sub(r'\s+', ' ', candidate_text).strip()
            norm_candidate = candidate_text.lower()
            
            # Ensure not empty and does not collide with validation or test sets
            if candidate_text and norm_candidate not in forbidden_eval_texts:
                augmented_text = candidate_text
                break
                
        tokens = re.findall(r"[a-z0-9_\[\]]+(?:[./:-][a-z0-9_\[\]]+)*", augmented_text.lower())
        tokenized_text = " ".join(tokens)
        
        record = {
            'Cleaned_Text': augmented_text,
            'Labels': original_row['Labels'],
            'Label_Count': original_row['Label_Count'],
            'Tokenized_Text': tokenized_text,
            'source_sample_id': original_row['source_sample_id'],
            'is_augmented': 1
        }
        augmented_records.append(record)
        
    df_augmented = pd.DataFrame(augmented_records)
    df_new_train = pd.concat([df_train.drop(columns=['Label_List']), df_augmented], ignore_index=True)
    
    final_all_labels = [lbl for sublist in df_new_train['Labels'].apply(lambda x: str(x).split(',')) for lbl in sublist]
    final_label_counts = Counter(final_all_labels)
    final_minority_classes = {lbl: count for lbl, count in final_label_counts.items() if count < target_count}
    
    print("\n" + "=" * 50)
    print("=== GENERIC WORDNET EDA AUGMENTATION REPORT ===")
    print(f"Original train size: {len(df_train):,} samples")
    print(f"Augmented train size: {len(df_new_train):,} samples (New samples: {len(df_augmented):,})")
    print(f"Original minority classes (< {target_count}): {len(minority_classes)}")
    print(f"Remaining minority classes (< {target_count}): {len(final_minority_classes)}")
    print("=" * 50)
    
    if save_csv:
        output_path = train_path.parent / output_filename
        df_new_train.to_csv(output_path, index=False, encoding='utf-8')
        print(f"[SUCCESS] Saved Generic EDA dataset to: {output_path}")
        
    return df_new_train


def run_all_generic_benchmarks(
    target_dataset='all',
    target_count=0,
    alpha_sr=0.13,
    alpha_ri=0.13,
    alpha_rs=0.13,
    p_rd=0.13,
    seed=42,
    save_csv=True,
    output_filename="train_augmented_generic_eda.csv"
):
    """
    Executes Generic WordNet EDA across 'cti_to_mitre', 'tram', 'joint' benchmarks.
    """
    if target_dataset == 'all':
        benchmarks = ['cti_to_mitre', 'tram', 'joint']
    elif target_dataset in ['joint', 'cti_to_mitre', 'tram']:
        benchmarks = [target_dataset]
    else:
        raise ValueError(f"Unknown target_dataset: '{target_dataset}'. Choose from ['all', 'joint', 'cti_to_mitre', 'tram']")
        
    for subset in benchmarks:
        train_path = Path(f"dataset/processed/{subset}/train.csv")
        if not train_path.exists():
            print(f"[WARNING] Cannot find {train_path}. Skipping '{subset}'.")
            continue
        print(f"\n>>> Running Generic EDA on Benchmark: [{subset.upper()}] <<<")
        run_generic_eda_for_benchmark(
            train_file=str(train_path),
            target_count=target_count,
            alpha_sr=alpha_sr,
            alpha_ri=alpha_ri,
            alpha_rs=alpha_rs,
            p_rd=p_rd,
            seed=seed,
            save_csv=save_csv,
            output_filename=output_filename
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generic WordNet EDA Pipeline for CTI Benchmarks")
    parser.add_argument(
        '--target_dataset',
        type=str,
        default='all',
        choices=['all', 'joint', 'cti_to_mitre', 'tram'],
        help='Target dataset benchmark to run (default: all)'
    )
    parser.add_argument('--train_file', type=str, default=None, help='Explicit path to input train.csv')
    parser.add_argument('--target_count', type=int, default=0, help='Minimum sample count target per class (0 = Auto-resolve to MEAN)')
    parser.add_argument('--alpha_sr', type=float, default=0.13, help='Synonym Replacement ratio (default: 0.13)')
    parser.add_argument('--alpha_ri', type=float, default=0.13, help='Random Insertion ratio (default: 0.13)')
    parser.add_argument('--alpha_rs', type=float, default=0.13, help='Random Swap ratio (default: 0.13)')
    parser.add_argument('--p_rd', type=float, default=0.13, help='Random Deletion probability (default: 0.13)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output_filename', type=str, default="train_augmented_generic_eda.csv", help='Output CSV filename')
    parser.add_argument('--no_save', action='store_true', help='Dry-run mode without saving CSV to disk')
    
    args = parser.parse_args()
    save_csv = not args.no_save
    
    if args.train_file is not None:
        run_generic_eda_for_benchmark(
            train_file=args.train_file,
            target_count=args.target_count,
            alpha_sr=args.alpha_sr,
            alpha_ri=args.alpha_ri,
            alpha_rs=args.alpha_rs,
            p_rd=args.p_rd,
            seed=args.seed,
            save_csv=save_csv,
            output_filename=args.output_filename
        )
    else:
        run_all_generic_benchmarks(
            target_dataset=args.target_dataset,
            target_count=args.target_count,
            alpha_sr=args.alpha_sr,
            alpha_ri=args.alpha_ri,
            alpha_rs=args.alpha_rs,
            p_rd=args.p_rd,
            seed=args.seed,
            save_csv=save_csv,
            output_filename=args.output_filename
        )

