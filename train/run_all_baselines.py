#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Baseline Evaluation Script

Retrains all baselines (RF+HMM, ESN, Mamba Smoother) and evaluates them
with the unified evaluation harness for fair comparison with P1.
"""

import argparse
import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation framework
spec = importlib.util.spec_from_file_location("evaluate_sequence_labeling", 
    project_root / "evaluation" / "evaluate_sequence_labeling.py")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)
create_evaluator = eval_module.create_evaluator

from classifier import Classifier
from utils import ordered_unique


def main():
    parser = argparse.ArgumentParser(description='Unified Baseline Evaluation')
    parser.add_argument('--data_dir', type=str, default='prepared_data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='artifacts/baselines',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("  UNIFIED BASELINE EVALUATION")
    print("="*70 + "\n")
    
    # Load data
    print("[*] Loading data...")
    data_dir = Path(args.data_dir)
    X_feats = pd.read_pickle(data_dir / 'X_feats.pkl').values
    Y = np.load(data_dir / 'Y_Walmsley2020.npy')
    P = np.load(data_dir / 'P.npy')
    
    print(f"  X shape: {X_feats.shape}")
    print(f"  Y shape: {Y.shape}, unique values: {np.unique(Y)}")
    print(f"  P shape: {P.shape}, unique participants: {len(np.unique(P))}")
    
    # Convert string labels to indices
    unique_labels = np.unique(Y)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    Y_idx = np.array([label_to_idx[label] for label in Y])
    print(f"  Label mapping: {label_to_idx}")
    
    # Split data (same as P1)
    print("\n[*] Splitting data (participant-level)...")
    unique_participants = sorted(np.unique(P))
    n_total = len(unique_participants)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_participants = unique_participants[:n_train]
    val_participants = unique_participants[n_train:n_train + n_val]
    test_participants = unique_participants[n_train + n_val:]
    
    train_mask = np.isin(P, train_participants)
    val_mask = np.isin(P, val_participants)
    test_mask = np.isin(P, test_participants)
    
    X_train, y_train, P_train = X_feats[train_mask], Y_idx[train_mask], P[train_mask]
    X_test, y_test, P_test = X_feats[test_mask], Y_idx[test_mask], P[test_mask]
    
    print(f"  Train: {X_train.shape[0]:6d} samples, {len(train_participants):3d} participants")
    print(f"  Test:  {X_test.shape[0]:6d} samples, {len(test_participants):3d} participants")
    
    # Initialize evaluator
    evaluator = create_evaluator(n_classes=len(unique_labels))
    
    # Store all results
    all_results = {}
    
    # Baseline B1: RF + HMM
    print("\n" + "="*70)
    print("  BASELINE B1: RF + HMM")
    print("="*70)
    
    model_rf_hmm = Classifier('rf_hmm', seed=args.seed)
    print("[*] Training RF+HMM...")
    model_rf_hmm.fit(X_train, y_train, P_train)
    
    print("[*] Evaluating RF+HMM...")
    y_pred_raw = model_rf_hmm.window_classifier.predict(X_test)
    y_pred_decoded = model_rf_hmm.predict(X_test, P_test)
    
    results_rf_hmm = evaluator.evaluate(y_test, y_pred_raw, y_pred_decoded, P_test,
                                        return_predictions=True)
    evaluator.print_results(results_rf_hmm, "RF+HMM Results")
    evaluator.save_results(results_rf_hmm, output_dir / 'rf_hmm')
    all_results['rf_hmm'] = results_rf_hmm
    
    # Baseline B2: RF only (no HMM)
    print("\n" + "="*70)
    print("  BASELINE B0: RF (no HMM)")
    print("="*70)
    
    model_rf = Classifier('rf', seed=args.seed)
    print("[*] Training RF...")
    model_rf.fit(X_train, y_train)
    
    print("[*] Evaluating RF...")
    y_pred_rf = model_rf.predict(X_test)
    
    results_rf = evaluator.evaluate(y_test, y_pred_rf, None, None,
                                    return_predictions=False)
    evaluator.print_results(results_rf, "RF (Raw) Results")
    evaluator.save_results(results_rf, output_dir / 'rf')
    all_results['rf'] = results_rf
    
    # Summary comparison
    print("\n" + "="*70)
    print("  BASELINE COMPARISON (Test Set Macro F1)")
    print("="*70)
    print(f"  RF:         {results_rf['raw']['macro_f1']:.4f} (raw)")
    print(f"  RF+HMM:     {results_rf_hmm['raw']['macro_f1']:.4f} (raw) / {results_rf_hmm['decoded']['macro_f1']:.4f} (decoded)")
    print(f"\n  [INFO] These are the apples-to-apples baselines for P1 comparison")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'split_info': {
            'n_train_participants': len(train_participants),
            'n_test_participants': len(test_participants),
            'n_train_samples': X_train.shape[0],
            'n_test_samples': X_test.shape[0],
        },
        'label_mapping': label_to_idx,
        'baseline_results': {
            'rf': {'raw_macro_f1': float(results_rf['raw']['macro_f1'])},
            'rf_hmm': {
                'raw_macro_f1': float(results_rf_hmm['raw']['macro_f1']),
                'decoded_macro_f1': float(results_rf_hmm['decoded']['macro_f1']),
            }
        },
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Baseline evaluation complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
