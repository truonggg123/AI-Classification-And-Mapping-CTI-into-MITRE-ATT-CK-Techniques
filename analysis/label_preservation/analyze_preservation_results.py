"""
Script: analyze_preservation_results.py
Description: Computes Inter-Annotator Agreement (Cohen's Kappa), observed agreement,
             and final consensus label-preservation metrics for CyberEDA and GenericEDA.
Author: Research Team
Date: 2026-10-10
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

SCRIPT_DIR = Path(__file__).resolve().parent
CYBER_CSV = SCRIPT_DIR / 'eval_label_preservation_cyber.csv'
GENERIC_CSV = SCRIPT_DIR / 'eval_label_preservation_generic.csv'

def evaluate_method(csv_file: Path, method_name: str, a1_name: str, a2_name: str):
    """
    Evaluates manual audit results for a given data augmentation method.
    """
    print("=" * 75)
    print(f"AUDIT EVALUATION REPORT: {method_name.upper()}")
    print("=" * 75)
    
    if not csv_file.exists():
        print(f"[ERROR] Evaluation file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    n_total = len(df)
    
    col_a1 = f'Label_Preserved_{a1_name} (Y/N)'
    col_a2 = f'Label_Preserved_{a2_name} (Y/N)'
    col_sem1 = f'Semantic_Preserved_{a1_name} (Y/P/N)'
    col_sem2 = f'Semantic_Preserved_{a2_name} (Y/P/N)'
    col_gram1 = f'Grammar_OK_{a1_name} (Y/N)'
    col_gram2 = f'Grammar_OK_{a2_name} (Y/N)'
    col_final = 'Final_Consensus_Label (Y/N)'
    
    # 1. Inter-Annotator Agreement between independent evaluators
    s1 = df[col_a1].astype(str).str.strip().str.upper()
    s2 = df[col_a2].astype(str).str.strip().str.upper()
    
    valid_mask = s1.isin(['Y', 'N']) & s2.isin(['Y', 'N'])
    n_annotated = valid_mask.sum()
    
    kappa = 0.0
    raw_agreement = 0.0
    if n_annotated > 0:
        kappa = cohen_kappa_score(s1[valid_mask], s2[valid_mask])
        raw_agreement = (s1[valid_mask] == s2[valid_mask]).mean() * 100
        disagreements = (s1[valid_mask] != s2[valid_mask]).sum()
        
        print(f"1. INTER-ANNOTATOR AGREEMENT ({a1_name} vs {a2_name}):")
        print(f"   - Annotated instances         : {n_annotated}/{n_total}")
        print(f"   - Observed Agreement          : {raw_agreement:.2f}%")
        print(f"   - Cohen's Kappa (κ)           : {kappa:.3f}")
        print(f"   - Disagreement count          : {disagreements} instances")
    else:
        print(f"[PENDING] Awaiting independent annotations for {a1_name} and {a2_name}.")
    
    # 2. Final Consensus Evaluation (Adjudicated by Arbiter / Lead Annotator)
    s_final = df[col_final].astype(str).str.strip().str.upper()
    final_valid = s_final.isin(['Y', 'N'])
    n_final = final_valid.sum()
    
    preservation_rate = 0.0
    drift_rate = 0.0
    if n_final > 0:
        n_preserved = (s_final[final_valid] == 'Y').sum()
        n_drift = (s_final[final_valid] == 'N').sum()
        preservation_rate = (n_preserved / n_final) * 100
        drift_rate = (n_drift / n_final) * 100
        
        print(f"\n2. FINAL CONSENSUS METRICS (ARBITER ADJUDICATED):")
        print(f"   - Adjudicated instances       : {n_final}/{n_total}")
        print(f"   - Label Preservation Rate (%) : {preservation_rate:.2f}% ({n_preserved}/{n_final})")
        print(f"   - Label Drift Rate (%)        : {drift_rate:.2f}% ({n_drift}/{n_final})")
        
        # Analyze error correlation with Random Deletion (RD)
        drift_rows = df[final_valid & (s_final == 'N')]
        rd_impact = drift_rows['Words_Removed_RD'].str.strip() != '(None)'
        n_rd_drift = rd_impact.sum()
        print(f"   - Label drift caused by RD    : {n_rd_drift}/{n_drift} ({n_rd_drift/max(1,n_drift)*100:.1f}%)")
    else:
        print("\n[PENDING] Final_Consensus_Label has not been completed by Arbiter yet.")
        
    print()
    return {
        'Method': method_name,
        'Annotators': f"{a1_name} & {a2_name}",
        'Evaluated': n_final,
        'Kappa': kappa,
        'Observed_Agreement_Pct': raw_agreement,
        'Label_Preserved_Pct': preservation_rate
    }

if __name__ == '__main__':
    res_cyber = evaluate_method(CYBER_CSV, 'CyberEDA', 'Annotator_1', 'Annotator_2')
    res_generic = evaluate_method(GENERIC_CSV, 'GenericEDA', 'Annotator_3', 'Annotator_4')
