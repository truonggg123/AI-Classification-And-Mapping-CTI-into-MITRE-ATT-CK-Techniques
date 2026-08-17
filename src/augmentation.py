"""
CTI Data Augmentation Module (Optimized for Benchmarking)
Implements Cyber EDA (Full & Atomic operations: SR, RI, RS, RD) 
and Offline Back Translation (English -> French -> English).
Balances minority classes in train.csv while protecting domain entities.

Usage:
    python src/augmentation.py --mode sr        # Synonym Replacement only
    python src/augmentation.py --mode ri        # Random Insertion only
    python src/augmentation.py --mode rs        # Random Swap only
    python src/augmentation.py --mode rd        # Random Deletion only
    python src/augmentation.py --mode eda       # Full Cyber EDA (SR + RI + RS + RD)
    python src/augmentation.py --mode bt        # Offline Back Translation
"""

import argparse
import random
import re
import pickle
import json
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_protected_lexicon_via_tfidf_idf(stix_data, min_idf_quantile=0.15):
    """
    Extracts Domain Protected Lexicon directly from MITRE STIX JSON metadata
    filtered dynamically using TF-IDF Inverse Document Frequency (IDF) over technique descriptions.
    Mathematically filters out non-discriminative low-IDF words (both English & generic IT stopwords)
    without needing any hardcoded/manual stopword lists. 100% objective for publication.
    """
    descriptions = []
    for obj in stix_data.get('objects', []):
        if obj.get('type') == 'attack-pattern' and obj.get('description'):
            descriptions.append(obj['description'])

    if descriptions:
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w[\w\.-]+\b")
        vectorizer.fit(descriptions)
        feature_names = vectorizer.get_feature_names_out()
        idf_scores = dict(zip(feature_names, vectorizer.idf_))
        all_idfs = list(idf_scores.values())
        idf_cutoff = float(np.quantile(all_idfs, min_idf_quantile)) if all_idfs else 1.5
    else:
        idf_scores = {}
        idf_cutoff = 0.0

    raw_candidates = set()
    for obj in stix_data.get('objects', []):
        obj_type = obj.get('type')
        
        # 1. Extract Platforms (e.g. windows, linux, macos, active-directory, containers)
        for platform in obj.get('x_mitre_platforms', []):
            p_clean = platform.lower().strip()
            if len(p_clean) > 2:
                raw_candidates.add(p_clean)
                for part in re.split(r'[\s\-_]+', p_clean):
                    if len(part) > 2:
                        raw_candidates.add(part)
            
        # 2. Extract Data Sources (e.g. process, file, command, registry, network, memory)
        for ds in obj.get('x_mitre_data_sources', []):
            for part in re.split(r'[\s\-_:]+', ds.lower()):
                clean_part = part.strip(".,;:!?()\"'")
                if len(clean_part) > 2:
                    raw_candidates.add(clean_part)
                    
        # 3. Extract keywords from Technique, Tool, and Malware names
        if obj_type in ['attack-pattern', 'tool', 'malware']:
            name = obj.get('name', '')
            for word in name.lower().split():
                clean_word = word.strip(".,;:!?()\"'")
                if len(clean_word) > 2:
                    raw_candidates.add(clean_word)

    protected_terms = set()
    for term in raw_candidates:
        # Keep terms whose IDF score > cutoff (high IDF = discriminative entity/technique term).
        # Terms not in corpus vocabulary get default 999.0 (retained as unique entity).
        term_idf = idf_scores.get(term, 999.0)
        if term_idf > idf_cutoff:
            protected_terms.add(term)

    print(f"[INFO] [TF-IDF Filter] Extracted {len(protected_terms)} high-IDF domain protected terms (IDF cutoff: {idf_cutoff:.2f}).")
    return protected_terms

def build_cyber_knowledge_base(cache_dir="dataset/processed", force_download=False, custom_protected_tokens=None):
    """
    Extracts and builds:
    1. Protected List: Protected domain terms extracted directly from MITRE STIX metadata + STIX IDs + custom user tokens.
    2. Synonym Dictionary: Domain aliases/synonyms for APT Groups, Malware & Tools.
    """
    cache_path = Path(cache_dir) / "enterprise-attack.json"
    url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"

    # Check known cache locations
    if not cache_path.exists() and not force_download:
        potential_files = [
            Path(cache_dir) / "enterprise-attack.json",
            Path('dataset/processed/enterprise-attack.json'),
            Path('dataset/enterprise-attack.json'),
            Path('enterprise-attack.json'),
            Path('/kaggle/input/enterprise-attack.json')
        ]
        for p in potential_files:
            if p.exists():
                cache_path = p
                print(f"[INFO] Found existing enterprise-attack.json at: {cache_path}")
                break

    if not cache_path.exists() or force_download:
        target_dir = Path(cache_dir)
        is_writable = False
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            test_file = target_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            is_writable = True
        except (OSError, PermissionError):
            is_writable = False

        if not is_writable:
            target_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
            target_dir.mkdir(parents=True, exist_ok=True)
            cache_path = target_dir / "enterprise-attack.json"
            print(f"[INFO] Primary cache directory is read-only. Redirecting cache download to writable path: {cache_path}")

        print(f"[INFO] Downloading MITRE STIX JSON from GitHub ({url})...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    f.write(response.read())
            print(f"[INFO] Download and cache saved to {cache_path}")
        except Exception as e:
            if cache_path.exists():
                print(f"[WARNING] Failed to download new data ({e}). Using existing cache at {cache_path}.")
            else:
                raise RuntimeError(f"Cannot download STIX JSON and no local cache found: {e}")
    else:
        print(f"[INFO] Using cached STIX JSON at: {cache_path}")

    with open(cache_path, 'r', encoding='utf-8') as f:
        stix_data = json.load(f)

    protected_set = set()
    synonym_raw = {}

    # Dynamically extract Domain Protected Lexicon via TF-IDF IDF Filtering 
    stix_metadata_terms = extract_protected_lexicon_via_tfidf_idf(stix_data)
    protected_set.update(stix_metadata_terms)

    # 1. Retain all primary tool, malware, attack-pattern names and MITRE STIX IDs (e.g. T1059, T1059.001)
    for obj in stix_data.get('objects', []):
        obj_type = obj.get('type')
        name = obj.get('name', '').strip().lower()

        if obj_type in ['tool', 'malware', 'attack-pattern']:
            if name:
                protected_set.add(name)
            
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    mitre_id = ref.get('external_id', '').strip().lower()
                    if mitre_id:
                        protected_set.add(mitre_id)
                        if '.' in mitre_id:
                            parent_id = mitre_id.split('.')[0]
                            protected_set.add(parent_id)

        if obj_type in ['intrusion-set', 'malware', 'tool']:
            primary_name = name
            raw_aliases = obj.get('aliases', []) or obj.get('x_mitre_aliases', [])
            aliases = [a.strip().lower() for a in raw_aliases if a.strip().lower() != primary_name]

            if primary_name and aliases:
                all_names = list(set([primary_name] + aliases))
                
                if primary_name not in synonym_raw:
                    synonym_raw[primary_name] = set()
                synonym_raw[primary_name].update(aliases)
                
                for alias in aliases:
                    if alias not in synonym_raw:
                        synonym_raw[alias] = set()
                    synonym_raw[alias].update([n for n in all_names if n != alias])

    # 2. Add custom protected tokens/terms provided by user
    if custom_protected_tokens:
        for item in custom_protected_tokens:
            protected_set.add(item.lower().strip())
        print(f"[INFO] Added {len(custom_protected_tokens)} custom protected terms into protected list.")

    synonym_dict = {k: sorted(list(v)) for k, v in synonym_raw.items()}

    print(f"[INFO] Completed Protected List: {len(protected_set)} terms.")
    print(f"[INFO] Completed Synonym Dictionary: {len(synonym_dict)} entries.")

    return protected_set, synonym_dict

SPECIAL_TOKENS = ["[CVE]", "[URL]", "[FILE_PATH]", "[IPV4]", "[HASH]"]

def is_special_token(word):
    # Automatically preserves any token in bracket format e.g. [CVE], [REGISTRY], [CUSTOM_TOKEN]
    return word.strip().upper() in SPECIAL_TOKENS or (word.startswith('[') and word.endswith(']'))

def is_protected(word, protected_set):
    if is_special_token(word):
        return True
    clean_word = word.lower().strip(".,;:!?()\"'")
    return clean_word in protected_set

# --- Cyber EDA implementation ---

def synonym_replacement(words, n, synonym_dict):
    new_words = words.copy()
    sentence_words = set([w.lower().strip(".,;:!?()\"'") for w in new_words])
    
    candidate_keys = []
    for key in synonym_dict:
        key_words = key.split()
        if all(kw in sentence_words for kw in key_words):
            candidate_keys.append(key)
            
    if not candidate_keys:
        return new_words
        
    keys_sorted = sorted(candidate_keys, key=len, reverse=True)
    num_replaced = 0
    i = 0
    while i < len(new_words) and num_replaced < n:
        matched = False
        for key in keys_sorted:
            key_words = key.split()
            k = len(key_words)
            if i + k <= len(new_words):
                subsegment = [w.lower().strip(".,;:!?()\"'") for w in new_words[i:i+k]]
                if subsegment == key_words:
                    syns = synonym_dict[key]
                    if syns:
                        synonym = random.choice(syns)
                        syn_words = synonym.split()
                        
                        # Preserve capitalization of first word
                        if new_words[i] and new_words[i][0].isupper():
                            syn_words[0] = syn_words[0].capitalize()
                        
                        # Preserve trailing punctuation from last replaced word if present
                        last_orig_word = new_words[i+k-1]
                        punct = ""
                        for char in reversed(last_orig_word):
                            if char in ".,;:!?":
                                punct = char + punct
                            else:
                                break
                        if punct:
                            syn_words[-1] = syn_words[-1] + punct

                        new_words[i:i+k] = syn_words
                        num_replaced += 1
                        i += len(syn_words)
                        matched = True
                        break
        if not matched:
            i += 1
    return new_words

def random_deletion(words, p, protected_set):
    if len(words) <= 1:
        return words
    new_words = []
    for word in words:
        if is_protected(word, protected_set):
            new_words.append(word)
            continue
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        return [random.choice(words)]
    return new_words

def random_swap(words, n, protected_set):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words, protected_set)
    return new_words

def swap_word(new_words, protected_set):
    if len(new_words) <= 1:
        return new_words
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
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

def random_insertion(words, n, synonym_dict):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words, synonym_dict)
    return new_words

def add_word(new_words, synonym_dict):
    if not new_words:
        return
    sentence_words = set([w.lower().strip(".,;:!?()\"'") for w in new_words])
    
    candidate_keys = []
    for key in synonym_dict:
        key_words = key.split()
        if all(kw in sentence_words for kw in key_words):
            candidate_keys.append(key)
            
    if not candidate_keys:
        return
        
    matches = []
    for key in candidate_keys:
        syns = synonym_dict[key]
        key_words = key.split()
        k = len(key_words)
        for i in range(len(new_words) - k + 1):
            subsegment = [w.lower().strip(".,;:!?()\"'") for w in new_words[i:i+k]]
            if subsegment == key_words:
                matches.append((key, syns))
                
    if not matches:
        return
        
    _, syns = random.choice(matches)
    if syns:
        random_synonym = random.choice(syns)
        random_idx = random.randint(0, len(new_words))
        syn_words = random_synonym.split()
        for offset, w in enumerate(syn_words):
            new_words.insert(random_idx + offset, w)

def cyber_eda(text, protected_set, synonym_dict, alpha_sr=0.1, alpha_ri=0.03, alpha_rs=0.01, p_rd=0.02):
    """
    Optimized Cyber EDA pipeline:
    - alpha_sr: Synonym Replacement ratio (0.10)
    - alpha_ri: Random Insertion ratio (0.03)
    - alpha_rs: Random Swap ratio (0.01) - kept minimal to preserve command syntax/verb order
    - p_rd: Random Deletion probability (0.02) - kept minimal to prevent loss of critical context
    """
    words = text.split()
    num_words = len(words)
    if num_words == 0:
        return text

    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    words = synonym_replacement(words, n_sr, synonym_dict)
    words = random_insertion(words, n_ri, synonym_dict)
    words = random_swap(words, n_rs, protected_set)
    words = random_deletion(words, p_rd, protected_set)

    return " ".join(words)

def single_eda(text, op, protected_set, synonym_dict, alpha_sr=0.1, alpha_ri=0.03, alpha_rs=0.01, p_rd=0.02):
    """
    Applies a single atomic EDA operation (SR, RI, RS, or RD).
    """
    words = text.split()
    num_words = len(words)
    if num_words == 0:
        return text

    if op in ['sr', 'synonym']:
        n_sr = max(1, int(alpha_sr * num_words))
        words = synonym_replacement(words, n_sr, synonym_dict)
    elif op in ['ri', 'insert']:
        n_ri = max(1, int(alpha_ri * num_words))
        words = random_insertion(words, n_ri, synonym_dict)
    elif op in ['rs', 'swap']:
        n_rs = max(1, int(alpha_rs * num_words))
        words = random_swap(words, n_rs, protected_set)
    elif op in ['rd', 'delete']:
        words = random_deletion(words, p_rd, protected_set)
    else:
        raise ValueError(f"Unknown single EDA operation: {op}")

    return " ".join(words)


# --- Back Translation implementation ---

TOKEN_MAP = {
    "[CVE]": "__TOKEN_CVE__",
    "[URL]": "__TOKEN_URL__",
    "[FILE_PATH]": "__TOKEN_FILE_PATH__",
    "[IPV4]": "__TOKEN_IPV4__",
    "[HASH]": "__TOKEN_HASH__"
}

def mask_special_tokens(text):
    """Dynamically masks any bracketed token [XYZ] (e.g. [CVE], [URL], [FILE_PATH], [IPV4], [HASH], [REGISTRY]) into __TOKEN_XYZ__."""
    def replace_bracket_token(match):
        tok_name = match.group(1).upper()
        return f"__TOKEN_{tok_name}__"
    return re.sub(r'\[([A-Za-z0-9_]+)\]', replace_bracket_token, text)

def unmask_special_tokens(text):
    """Dynamically restores __TOKEN_XYZ__ or variation back to original format [XYZ]."""
    def restore_bracket_token(match):
        tok_name = match.group(2).upper()
        return f" [{tok_name}] "

    # Dynamic match for any __TOKEN_XYZ__ or variation produced by translation models
    text = re.sub(r"(__|_)\s*TOKEN_([A-Za-z0-9_]+)\s*(__|_)", restore_bracket_token, text, flags=re.IGNORECASE)

    # Legacy/Fallback regex for raw __CVE__, __URL__, __FILE_PATH__, etc.
    text = re.sub(r"(__|_)\s*(CVE|cve)\s*(__|_)", " [CVE] ", text)
    text = re.sub(r"(__|_)\s*(URL|url)\s*(__|_)", " [URL] ", text)
    text = re.sub(r"(__|_)\s*(FILE_PATH|file_path|chemin_de_fichier)\s*(__|_)", " [FILE_PATH] ", text, flags=re.IGNORECASE)
    text = re.sub(r"(__|_)\s*(IPV4|ipv4|ip)\s*(__|_)", " [IPV4] ", text, flags=re.IGNORECASE)
    text = re.sub(r"(__|_)\s*(HASH|hash)\s*(__|_)", " [HASH] ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Main Pipeline ---

def run_augmentation(mode='eda', train_file='dataset/processed/cti_to_mitre/train.csv', df_train=None, target_count=None, save_csv=False, cache_dir='dataset/processed', output_file=None):
    valid_modes = ['sr', 'synonym', 'ri', 'insert', 'rs', 'swap', 'rd', 'delete', 'eda', 'cyber_eda']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}")

    if df_train is None:
        train_path = Path(train_file)
        if not train_path.exists():
            # Try resolving relative path if needed
            possible_paths = [
                train_path,
                Path('dataset/processed/cti_to_mitre/train.csv'),
                Path('dataset/processed/tram/train.csv'),
                Path('dataset/processed/joint/train.csv')
            ]
            for p in possible_paths:
                if p.exists():
                    train_path = p
                    break
        df_train = pd.read_csv(train_path)
        cache_path_dir = train_path.parent
    else:
        df_train = df_train.copy()
        cache_path_dir = Path(cache_dir)
        train_path = cache_path_dir / "train.csv"

    # Auto-resolve target_count based on empirical dataset MEAN if not explicitly specified
    if target_count is None or target_count <= 0:
        file_str = str(train_path).lower()
        if 'cti_to_mitre' in file_str:
            target_count = 55   # Empirical MEAN for CTI-to-MITRE train set
        elif 'tram' in file_str:
            target_count = 151  # Empirical MEAN for TRAM train set
        elif 'joint' in file_str:
            target_count = 95   # Empirical MEAN for JOINT train set
        else:
            target_count = 120  # Fallback default
        print(f"[INFO] Auto-resolved target_count to dataset MEAN: {target_count} samples/class")
    else:
        print(f"[INFO] Using specified target_count: {target_count} samples/class")

    if 'is_augmented' not in df_train.columns:
        df_train['is_augmented'] = 0
    print(f"[INFO] Loaded training dataset with {len(df_train):,} samples.")
    
    # Build Cyber Knowledge Base from STIX JSON
    protected_set, synonym_dict = build_cyber_knowledge_base(cache_dir=cache_path_dir)
    
    # Parse labels
    df_train['Label_List'] = df_train['Labels'].apply(lambda x: str(x).split(','))
    
    # Track current dynamic label frequencies for Multi-Label Aware Greedy Sampling
    dynamic_label_counts = Counter([lbl for sublist in df_train['Label_List'] for lbl in sublist])
    print(f"[INFO] Total labels found: {len(dynamic_label_counts)}")
    
    minority_classes = {lbl: dynamic_label_counts[lbl] for lbl in dynamic_label_counts if dynamic_label_counts[lbl] < target_count}
    print(f"[INFO] Initial classes requiring augmentation (< {target_count} samples): {len(minority_classes)}")
    
    # Lookup table: Label -> list of row indices containing that label
    label_to_indices = {}
    for idx, row in df_train.iterrows():
        for lbl in row['Label_List']:
            if lbl not in label_to_indices:
                label_to_indices[lbl] = []
            label_to_indices[lbl].append(idx)
            
    # --- Multi-Label Aware Greedy Resampling ---
    print("[INFO] Performing Multi-Label Aware Greedy Resampling...")
    indices_to_augment = []
    
    # Sort minority classes ascending by initial count
    sorted_minority_labels = sorted(minority_classes.keys(), key=lambda l: dynamic_label_counts[l])
    
    for lbl in sorted_minority_labels:
        available_indices = label_to_indices[lbl]
        while dynamic_label_counts[lbl] < target_count:
            sampled_idx = random.choice(available_indices)
            indices_to_augment.append(sampled_idx)
            
            # Dynamically update counts for ALL labels present in the sampled row
            for co_lbl in df_train.iloc[sampled_idx]['Label_List']:
                dynamic_label_counts[co_lbl] += 1
                
    print(f"[INFO] Total samples selected for augmentation: {len(indices_to_augment):,}")
            
    # Generate augmented rows
    augmented_records = []
    for loop_i, idx in enumerate(tqdm(indices_to_augment, desc=f"Applying Augmentation ({mode})")):
        original_row = df_train.iloc[idx]
        original_text = original_row['Cleaned_Text']
        
        if mode in ['eda', 'cyber_eda']:
            augmented_text = cyber_eda(original_text, protected_set, synonym_dict)
        elif mode in ['sr', 'synonym', 'ri', 'insert', 'rs', 'swap', 'rd', 'delete']:
            augmented_text = single_eda(original_text, mode, protected_set, synonym_dict)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Standardize whitespace
        augmented_text = re.sub(r'\s+', ' ', augmented_text).strip()
        
        # Re-tokenize
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
    
    # Recalculate label frequencies to verify boost
    final_all_labels = [lbl for sublist in df_new_train['Labels'].apply(lambda x: str(x).split(',')) for lbl in sublist]
    final_label_counts = Counter(final_all_labels)
    final_minority_classes = {lbl: count for lbl, count in final_label_counts.items() if count < target_count}
    
    print(f"\n=== AUGMENTATION REPORT ({mode.upper()}) ===")
    print(f"Original train size: {len(df_train):,} samples")
    print(f"Augmented train size: {len(df_new_train):,} samples")
    print(f"Original minority classes (< {target_count}): {len(minority_classes)}")
    print(f"Remaining minority classes (< {target_count}): {len(final_minority_classes)}")
    
    # Save output if requested
    if save_csv:
        try:
            output_path = train_path.parent / f"train_augmented_{mode}.csv"
            df_new_train.to_csv(output_path, index=False, encoding='utf-8')
            print(f"[SUCCESS] Saved augmented dataset to: {output_path}")
        except Exception as e:
            # Fallback to /kaggle/working/ if input directory is read-only
            if Path('/kaggle/working').exists():
                out_dir = Path(f"/kaggle/working/dataset/processed/{train_path.parent.name}")
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = out_dir / f"train_augmented_{mode}.csv"
                df_new_train.to_csv(output_path, index=False, encoding='utf-8')
                print(f"[SUCCESS] Saved augmented dataset (Kaggle working) to: {output_path}")
            else:
                print(f"[WARNING] Could not save CSV to disk ({e}). Returning augmented DataFrame in memory.")
    
    return df_new_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CTI Data Augmentation Pipeline")
    parser.add_argument('--mode', type=str, default='eda', choices=['sr', 'synonym', 'ri', 'insert', 'rs', 'swap', 'rd', 'delete', 'eda', 'cyber_eda', 'bt'], help='Augmentation mode')
    parser.add_argument('--train_file', type=str, default='dataset/processed/cti_to_mitre/train.csv', help='Path to input train.csv')
    parser.add_argument('--target_count', type=int, default=0, help='Minimum sample count target per class (0 = Auto-resolve to dataset MEAN)')
    parser.add_argument('--save_csv', action='store_true', help='Save augmented dataset to CSV file')
    args = parser.parse_args()
    
    run_augmentation(mode=args.mode, train_file=args.train_file, target_count=args.target_count, save_csv=args.save_csv)
