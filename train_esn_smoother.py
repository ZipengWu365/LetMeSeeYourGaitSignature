#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESN Smoother Training Script - CUDA Accelerated Version
Uses Echo State Network for temporal smoothing with GPU acceleration
"""

import argparse
import sys
import traceback
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  [OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  [WARN] CUDA not available, using CPU")
    return device


class ESNSmootherWrapper:
    """sklearn-style wrapper for ESN Smoother with CUDA acceleration"""

    def __init__(self, n_classes=4, n_reservoir=800, spectral_radius=0.9,
                 input_scaling=0.5, sparsity=0.1, ridge_alpha=1.0, device=None):
        self.n_classes = n_classes
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.ridge_alpha = ridge_alpha
        self.device = device or get_device()
        self.label_encoder = None

        # Initialize weights
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.n_input = None

    def _initialize_weights(self, n_input):
        """Initialize ESN weights on GPU"""
        self.n_input = n_input

        # Input weights
        self.W_in = torch.randn(n_input, self.n_reservoir, device=self.device)
        self.W_in = self.W_in * self.input_scaling

        # Reservoir weights (sparse)
        W_res = torch.randn(self.n_reservoir, self.n_reservoir, device=self.device)
        mask = torch.rand(self.n_reservoir, self.n_reservoir, device=self.device) > self.sparsity
        W_res[mask] = 0

        # Scale to spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        max_eigenvalue = torch.max(torch.abs(eigenvalues)).item()
        if max_eigenvalue > 0:
            W_res = W_res * (self.spectral_radius / max_eigenvalue)
        self.W_res = W_res

    def _run_reservoir(self, inputs):
        """Run reservoir with inputs - GPU accelerated"""
        seq_len = inputs.shape[0]
        states = torch.zeros(seq_len, self.n_reservoir, device=self.device)
        h = torch.zeros(self.n_reservoir, device=self.device)

        for t in range(seq_len):
            x_t = inputs[t]
            h = torch.tanh(x_t @ self.W_in + h @ self.W_res)
            states[t] = h

        return states

    def fit(self, y_pred_proba, y_true, groups, aux_features=None):
        """Train ESN smoother"""
        print(f"\n[*] Training ESN Smoother (CUDA: {self.device.type == 'cuda'})")

        # Prepare inputs
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba

        n_input = inputs.shape[1]
        print(f"  Input dimension: {n_input}")
        print(f"  Reservoir size: {self.n_reservoir}")

        # Initialize weights
        print("  Initializing reservoir...")
        self._initialize_weights(n_input)

        # Process each sequence (by participant)
        print("  Running reservoir on sequences...")
        all_states = []
        all_targets = []
        unique_groups = np.unique(groups)

        for i, g in enumerate(unique_groups):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(unique_groups)}")

            mask = groups == g
            seq_inputs = torch.FloatTensor(inputs[mask]).to(self.device)
            seq_targets = y_true[mask]

            # Run reservoir (GPU accelerated)
            states = self._run_reservoir(seq_inputs)

            # Move back to CPU for Ridge regression
            all_states.append(states.cpu().numpy())
            all_targets.append(seq_targets)

        # Combine all states
        X_states = np.vstack(all_states)
        y_targets = np.concatenate(all_targets)

        # Encode labels (handle string labels like 'sleep', 'walking', etc.)
        if y_targets.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            print("  [INFO] Detected string labels, using LabelEncoder")
            self.label_encoder = LabelEncoder()
            y_targets_encoded = self.label_encoder.fit_transform(y_targets)
            print(f"  [INFO] Classes: {self.label_encoder.classes_}")
        else:
            y_targets_encoded = y_targets.astype(np.int64)

        print(f"  Reservoir states: {X_states.shape}")

        # Train output layer using Ridge regression (GPU accelerated)
        print("  Training output layer (Ridge regression)...")

        # PyTorch Ridge regression (GPU accelerated)
        X_tensor = torch.FloatTensor(X_states).to(self.device)

        # One-hot encode targets
        y_onehot = np.zeros((len(y_targets), self.n_classes))
        y_onehot[np.arange(len(y_targets)), y_targets_encoded] = 1
        y_tensor = torch.FloatTensor(y_onehot).to(self.device)

        # Ridge regression closed-form solution: W = (X^T X + alpha*I)^{-1} X^T y
        XtX = X_tensor.T @ X_tensor
        XtX += self.ridge_alpha * torch.eye(self.n_reservoir, device=self.device)
        Xty = X_tensor.T @ y_tensor

        self.W_out = torch.linalg.solve(XtX, Xty)

        print("  [OK] Training complete")

    def predict(self, y_pred_proba, groups, aux_features=None):
        """Predict with trained model - GPU accelerated"""
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba

        all_preds = []
        unique_groups = np.unique(groups)

        for g in unique_groups:
            mask = groups == g
            seq_inputs = torch.FloatTensor(inputs[mask]).to(self.device)

            # Run reservoir
            states = self._run_reservoir(seq_inputs)

            # Compute output
            logits = states @ self.W_out
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.append((mask, preds))

        # Reconstruct predictions
        y_pred = np.zeros(len(groups), dtype=np.int64)
        for mask, preds in all_preds:
            y_pred[mask] = preds

        # Convert back to original labels if using LabelEncoder
        if self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred)

        return y_pred


def compute_auxiliary_features(X_raw, mode='enmo'):
    """Compute auxiliary features"""
    if mode == 'none':
        return None
    elif mode == 'enmo':
        enmo = np.sqrt((X_raw**2).sum(axis=-1)).mean(axis=-1, keepdims=True)
        return enmo
    else:  # full
        enmo = np.sqrt((X_raw**2).sum(axis=-1))
        return np.column_stack([
            enmo.mean(axis=-1),
            enmo.std(axis=-1),
            enmo.max(axis=-1),
        ])


def main():
    parser = argparse.ArgumentParser(description='Train ESN Smoother (CUDA accelerated)')
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment ID')
    parser.add_argument('--n_reservoir', type=int, default=800, help='Reservoir size')
    parser.add_argument('--spectral_radius', type=float, default=0.9, help='Spectral radius')
    parser.add_argument('--aux_features', type=str, default='enmo',
                       choices=['none', 'enmo', 'full'], help='Auxiliary features')
    parser.add_argument('--data_dir', type=str, default='./prepared_data')
    parser.add_argument('--output_dir', type=str, default='./models')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Training ESN Smoother: {args.exp_id} (CUDA accelerated)")
    print(f"{'='*70}\n")

    try:
        device = get_device()

        # Load data
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("[1/4] Loading data...")
        
        # Check required files
        required_files = ['train_test_split.pkl', 'y_train_proba_rf.npy', 'y_test_proba_rf.npy', 
                          'Y_Walmsley2020.npy', 'P.npy']
        for f in required_files:
            fpath = data_dir / f
            if not fpath.exists():
                print(f"  [ERROR] Required file not found: {fpath}")
                print(f"  [INFO] Please run train_baseline.py first to generate RF probabilities")
                sys.exit(1)
            else:
                print(f"  [OK] Found: {f}")
        
        split_info = joblib.load(data_dir / 'train_test_split.pkl')
        train_mask = split_info['train_mask']
        test_mask = split_info['test_mask']

        y_train_proba = np.load(data_dir / 'y_train_proba_rf.npy')
        y_test_proba = np.load(data_dir / 'y_test_proba_rf.npy')

        Y = np.load(data_dir / 'Y_Walmsley2020.npy')
        P = np.load(data_dir / 'P.npy')

        y_train = Y[train_mask]
        y_test = Y[test_mask]
        P_train = P[train_mask]
        P_test = P[test_mask]

        print(f"  Training set: {y_train_proba.shape}")
        print(f"  Test set: {y_test_proba.shape}")
        print(f"  Labels dtype: {y_train.dtype}")

        # Auxiliary features
        aux_train = None
        aux_test = None
        if args.aux_features != 'none':
            print(f"\n[2/4] Computing auxiliary features ({args.aux_features})...")
            try:
                X_raw = np.load(data_dir / 'X.npy', mmap_mode='r')
                aux_train = compute_auxiliary_features(X_raw[train_mask], args.aux_features)
                aux_test = compute_auxiliary_features(X_raw[test_mask], args.aux_features)
                print(f"  Auxiliary feature dim: {aux_train.shape[1]}")
            except FileNotFoundError:
                print("  [WARN] X.npy not found, skipping auxiliary features")
        else:
            print("\n[2/4] Skipping auxiliary features...")

        # Training
        print(f"\n[3/4] Training ESN Smoother...")
        esn = ESNSmootherWrapper(
            n_classes=4,
            n_reservoir=args.n_reservoir,
            spectral_radius=args.spectral_radius,
            device=device,
        )

        esn.fit(y_train_proba, y_train, P_train, aux_train)

        # Evaluation
        print("\n[4/4] Evaluating model...")
        y_pred = esn.predict(y_test_proba, P_test, aux_test)

        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        print("\n" + "="*70)
        print(f"  {args.exp_id} Test Results")
        print("="*70)
        print(classification_report(y_test, y_pred))
        print(f"\nMacro F1: {f1_macro:.4f}")
        print(f"Per-class F1: {f1_per_class}")

        # Save model
        model_path = output_dir / f'esn_{args.exp_id}.pkl'
        joblib.dump(esn, model_path)
        print(f"\n[OK] Model saved: {model_path}")

        print(f"\n{'='*70}")
        print(f"  [OK] Training complete! Macro F1 = {f1_macro:.4f}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
