"""
CTI Data Augmentation Module
Implements Cyber EDA, Offline Back Translation (English -> French -> English),
and Hybrid (Back Translation + Cyber EDA) augmentation.
Balances the minority classes in train.csv to have at least 300 samples.

Usage:
    python src/augmentation.py --mode eda
    python src/augmentation.py --mode bt
    python src/augmentation.py --mode hybrid
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

def build_cyber_knowledge_base(cache_dir="dataset/processed", force_download=False):
    """
    Extracts and builds:
    1. Protected List: Protected domain terms (Tools, Malware, Techniques & IDs).
    2. Synonym Dictionary: Domain aliases/synonyms for APT Groups, Malware & Tools.

    Optimizations:
    - Extracts directly from raw MITRE STIX JSON without pyattck dependency.
    - Supports automatic offline caching to accelerate execution and enable offline runs.
    - Resolves alias conflicts/overwrites via Set Merging.
    - Automatically appends parent technique codes when sub-techniques are encountered.
    """
    cache_path = Path(cache_dir) / "enterprise-attack.json"
    url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"

    # Search for existing enterprise-attack.json in cache_dir, parent dirs, or kaggle input
    if not cache_path.exists() and not force_download:
        search_dirs = [Path(cache_dir), Path('.'), Path('..'), Path('/kaggle/input')]
        for d in search_dirs:
            if d.exists():
                matches = list(d.rglob('enterprise-attack.json'))
                if matches:
                    cache_path = matches[0]
                    print(f"[INFO] Found existing enterprise-attack.json at: {cache_path}")
                    break

    if not cache_path.exists() or force_download:
        # Determine writable cache directory (e.g. /kaggle/working if cache_dir is read-only)
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

    common_system_terms = {
        "powershell", "cmd.exe", "cmd", "bash", "registry", "dll", "exe",
        "bypass", "privilege", "escalation", "port", "http", "https", "ssh",
        "system32", "lsass.exe", "svchost.exe", "active directory"
    }
    protected_set.update(common_system_terms)

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

    synonym_dict = {k: sorted(list(v)) for k, v in synonym_raw.items()}

    print(f"[INFO] Completed Protected List: {len(protected_set)} terms.")
    print(f"[INFO] Completed Synonym Dictionary: {len(synonym_dict)} entries.")

    return protected_set, synonym_dict

SPECIAL_TOKENS = ["[CVE]", "[URL]", "[FILE_PATH]", "[IPV4]", "[HASH]"]

def is_special_token(word):
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
    
    # Pre-filter candidate keys to reduce runtime overhead
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
                        if new_words[i] and new_words[i][0].isupper():
                            syn_words[0] = syn_words[0].capitalize()
                        
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
        
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
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
    
    # Pre-filter candidate keys to reduce runtime overhead
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

def cyber_eda(text, protected_set, synonym_dict, alpha_sr=0.1, alpha_ri=0.05, alpha_rs=0.05, p_rd=0.05):
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


# --- Back Translation implementation ---

TOKEN_MAP = {
    "[CVE]": "__CVE__",
    "[URL]": "__URL__",
    "[FILE_PATH]": "__FILE_PATH__",
    "[IPV4]": "__IPV4__",
    "[HASH]": "__HASH__"
}

def mask_special_tokens(text):
    for tok, mask in TOKEN_MAP.items():
        text = text.replace(tok, mask)
    return text

def unmask_special_tokens(text):
    text = re.sub(r"__\s*CVE\s*__", " [CVE] ", text, flags=re.IGNORECASE)
    text = re.sub(r"__\s*URL\s*__", " [URL] ", text, flags=re.IGNORECASE)
    text = re.sub(r"__\s*FILE_PATH\s*__", " [FILE_PATH] ", text, flags=re.IGNORECASE)
    text = re.sub(r"__\s*IPV4\s*__", " [IPV4] ", text, flags=re.IGNORECASE)
    text = re.sub(r"__\s*HASH\s*__", " [HASH] ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class OfflineBackTranslator:
    def __init__(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing translation models on device: {self.device}")
        
        self.en_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
        self.fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
        
        self.en_fr_tok = MarianTokenizer.from_pretrained(self.en_fr_model_name)
        self.en_fr_model = MarianMTModel.from_pretrained(self.en_fr_model_name).to(self.device)
        
        self.fr_en_tok = MarianTokenizer.from_pretrained(self.fr_en_model_name)
        self.fr_en_model = MarianMTModel.from_pretrained(self.fr_en_model_name).to(self.device)
        
        self.cache = {}

    def translate_batch(self, texts):
        # Mask special tokens
        masked_texts = [mask_special_tokens(t) for t in texts]
        
        # Find unique texts not in cache
        unique_masked = list(set([t for t in masked_texts if t not in self.cache]))
        
        if unique_masked:
            print(f"[INFO] Translating {len(unique_masked)} unique segments...")
            import torch
            
            # Translate to French (chunk size 32)
            fr_texts = []
            batch_size = 32
            for i in range(0, len(unique_masked), batch_size):
                batch = unique_masked[i:i+batch_size]
                inputs = self.en_fr_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
                with torch.no_grad():
                    outputs = self.en_fr_model.generate(**inputs, num_beams=1, max_new_tokens=256)
                fr_texts.extend(self.en_fr_tok.batch_decode(outputs, skip_special_tokens=True))
                
            # Translate back to English
            en_texts = []
            for i in range(0, len(fr_texts), batch_size):
                batch = fr_texts[i:i+batch_size]
                inputs = self.fr_en_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
                with torch.no_grad():
                    outputs = self.fr_en_model.generate(**inputs, num_beams=1, max_new_tokens=256)
                en_texts.extend(self.fr_en_tok.batch_decode(outputs, skip_special_tokens=True))
            
            # Cache results
            for original_masked, back_translated_masked in zip(unique_masked, en_texts):
                self.cache[original_masked] = back_translated_masked

        # Retrieve and unmask
        results = []
        for mt in masked_texts:
            back_translated_masked = self.cache[mt]
            results.append(unmask_special_tokens(back_translated_masked))
            
        return results

# --- Main Pipeline ---

def run_augmentation(mode='eda', train_file='dataset/processed/train.csv', df_train=None, target_count=120, save_csv=False, cache_dir='dataset/processed'):
    if df_train is None:
        train_path = Path(train_file)
        df_train = pd.read_csv(train_path)
        cache_path_dir = train_path.parent
    else:
        df_train = df_train.copy()
        cache_path_dir = Path(cache_dir)
        train_path = cache_path_dir / "train.csv"

    if 'is_augmented' not in df_train.columns:
        df_train['is_augmented'] = 0
    print(f"[INFO] Loaded training dataset with {len(df_train):,} samples.")
    
    # Initialize and build Cyber Knowledge Base from STIX JSON
    protected_set, synonym_dict = build_cyber_knowledge_base(cache_dir=cache_path_dir)
    
    if mode == 'hybrid':
        bt_path = train_path.parent / "train_augmented_bt.csv"
        if bt_path.exists():
            print("[INFO] Found existing train_augmented_bt.csv. Using fast path for hybrid mode...")
            df_bt = pd.read_csv(bt_path)
            df_train_len = len(df_train)
            df_augmented = df_bt.iloc[df_train_len:].copy().reset_index(drop=True)
            
            print("[INFO] Applying Cyber EDA on back-translated samples...")
            tqdm.pandas(desc="Applying Cyber EDA")
            df_augmented['Cleaned_Text'] = df_augmented['Cleaned_Text'].progress_apply(
                lambda x: cyber_eda(x, protected_set, synonym_dict)
            )
            df_augmented['Cleaned_Text'] = df_augmented['Cleaned_Text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
            
            # Re-tokenize
            df_augmented['Tokenized_Text'] = df_augmented['Cleaned_Text'].apply(
                lambda x: " ".join(re.findall(r"[a-z0-9_\[\]]+(?:[./:-][a-z0-9_\[\]]+)*", str(x).lower()))
            )
            
            df_new_train = pd.concat([df_train, df_augmented], ignore_index=True)
            
            # Recalculate label frequencies to verify boost
            new_all_labels = [lbl for sublist in df_new_train['Labels'].apply(lambda x: str(x).split(',')) for lbl in sublist]
            new_label_counts = Counter(new_all_labels)
            new_minority_classes = {lbl: count for lbl, count in new_label_counts.items() if count < target_count}
            
            print(f"\n=== AUGMENTATION REPORT (HYBRID - FAST PATH) ===")
            print(f"Original train size: {len(df_train):,} samples")
            print(f"Augmented train size: {len(df_new_train):,} samples")
            print(f"New minority classes (< {target_count}): {len(new_minority_classes)}")
            
            # Save output if requested
            if save_csv:
                output_path = train_path.parent / f"train_augmented_hybrid.csv"
                df_new_train.to_csv(output_path, index=False, encoding='utf-8')
                print(f"[SUCCESS] Saved augmented dataset to: {output_path}")
            return df_new_train
    
    # Parse labels
    df_train['Label_List'] = df_train['Labels'].apply(lambda x: str(x).split(','))
    
    # Calculate label frequencies
    all_labels = [lbl for sublist in df_train['Label_List'] for lbl in sublist]
    label_counts = Counter(all_labels)
    print(f"[INFO] Total labels found: {len(label_counts)}")
    
    minority_classes = {lbl: count for lbl, count in label_counts.items() if count < target_count}
    print(f"[INFO] Classes requiring augmentation (< {target_count} samples): {len(minority_classes)}")
    
    # Build a lookup table from label -> row indices
    label_to_indices = {}
    for idx, row in df_train.iterrows():
        for lbl in row['Label_List']:
            if lbl not in label_to_indices:
                label_to_indices[lbl] = []
            label_to_indices[lbl].append(idx)
            
    # Sample new samples to augment
    new_rows = []
    print("[INFO] Selecting samples for minority class boosting...")
    
    # We store which indices we select for augmentation so we can process them
    indices_to_augment = []
    indices_to_label = [] # Keep track of labels for augmented rows
    
    for lbl, count in minority_classes.items():
        needed = target_count - count
        available_indices = label_to_indices[lbl]
        
        # Sample with replacement
        sampled_indices = random.choices(available_indices, k=needed)
        indices_to_augment.extend(sampled_indices)
        
    print(f"[INFO] Total samples selected for augmentation: {len(indices_to_augment):,}")
    
    # Perform translation first if needed
    translator = None
    if mode in ['bt', 'hybrid']:
        print("[INFO] Initializing offline translator...")
        translator = OfflineBackTranslator()
        
        # Gather unique texts to translate to avoid translating duplicates
        unique_indices = list(set(indices_to_augment))
        unique_texts = [df_train.iloc[idx]['Cleaned_Text'] for idx in unique_indices]
        
        # Batch translate all unique texts
        translated_unique = []
        batch_size = 128
        for i in tqdm(range(0, len(unique_texts), batch_size), desc="Back Translation"):
            batch = unique_texts[i:i+batch_size]
            translated_batch = translator.translate_batch(batch)
            translated_unique.extend(translated_batch)
            
        # Map back to original indices
        text_map = {idx: trans for idx, trans in zip(unique_indices, translated_unique)}
        
    # Generate augmented rows
    augmented_records = []
    for idx in tqdm(indices_to_augment, desc=f"Applying Augmentation ({mode})"):
        original_row = df_train.iloc[idx]
        original_text = original_row['Cleaned_Text']
        
        if mode == 'eda':
            augmented_text = cyber_eda(original_text, protected_set, synonym_dict)
        elif mode == 'bt':
            augmented_text = text_map[idx]
        elif mode == 'hybrid':
            translated_text = text_map[idx]
            augmented_text = cyber_eda(translated_text, protected_set, synonym_dict)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Standardize whitespace & lower for tokenization
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
    new_all_labels = [lbl for sublist in df_new_train['Labels'].apply(lambda x: str(x).split(',')) for lbl in sublist]
    new_label_counts = Counter(new_all_labels)
    new_minority_classes = {lbl: count for lbl, count in new_label_counts.items() if count < target_count}
    
    print(f"\n=== AUGMENTATION REPORT ({mode.upper()}) ===")
    print(f"Original train size: {len(df_train):,} samples")
    print(f"Augmented train size: {len(df_new_train):,} samples")
    print(f"Original minority classes (< {target_count}): {len(minority_classes)}")
    print(f"New minority classes (< {target_count}): {len(new_minority_classes)}")
    
    # Save output if requested
    if save_csv:
        output_path = train_path.parent / f"train_augmented_{mode}.csv"
        df_new_train.to_csv(output_path, index=False, encoding='utf-8')
        print(f"[SUCCESS] Saved augmented dataset to: {output_path}")
    
    return df_new_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CTI Data Augmentation Pipeline")
    parser.add_argument('--mode', type=str, default='eda', choices=['eda', 'bt', 'hybrid'], help='Augmentation mode')
    parser.add_argument('--train_file', type=str, default='dataset/processed/train.csv', help='Path to input train.csv')
    parser.add_argument('--target_count', type=int, default=120, help='Minimum sample count target per class')
    parser.add_argument('--save_csv', action='store_true', help='Save augmented dataset to CSV file')
    args = parser.parse_args()
    
    run_augmentation(mode=args.mode, train_file=args.train_file, target_count=args.target_count, save_csv=args.save_csv)
