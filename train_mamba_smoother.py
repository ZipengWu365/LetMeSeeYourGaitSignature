#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba Smoother Training Script - CUDA Accelerated Version
Uses Mamba SSM for temporal smoothing with GPU acceleration
"""

import argparse
import sys
import traceback
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Import PyTorch first (always needed)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"[ERROR] PyTorch not available: {e}")
    sys.exit(1)

# Try to import Mamba
print("[*] Importing Mamba...")
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("  [OK] Mamba-SSM available")
except ImportError as e:
    MAMBA_AVAILABLE = False
    print(f"  [WARN] Mamba-SSM not available: {e}")
    print("  [WARN] Will skip Mamba training")


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


class SequenceDataset(Dataset):
    """Sequence dataset grouped by participant"""
    def __init__(self, inputs, targets, groups, label_encoder=None):
        self.sequences = []
        self.targets_list = []
        self.label_encoder = label_encoder

        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            self.sequences.append(torch.FloatTensor(inputs[mask]))
            # Handle string labels by encoding them
            if label_encoder is not None:
                encoded_targets = label_encoder.transform(targets[mask])
            else:
                encoded_targets = targets[mask].astype(np.int64)
            self.targets_list.append(torch.LongTensor(encoded_targets))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets_list[idx]


def collate_sequences(batch):
    """Custom collate function for variable length sequences"""
    sequences, targets = zip(*batch)
    return sequences, targets


if MAMBA_AVAILABLE:
    class MambaSmoother(nn.Module):
        """Mamba Smoother for temporal smoothing - GPU accelerated"""

        def __init__(self, n_classes=4, d_model=64, n_layers=2, d_state=16, d_conv=4,
                     expand=2, dropout=0.1, aux_dim=0):
            super().__init__()
            self.n_classes = n_classes
            self.d_model = d_model

            input_dim = n_classes + aux_dim
            self.input_proj = nn.Linear(input_dim, d_model)

            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])

            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.output_proj = nn.Linear(d_model, n_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (batch, seq_len, input_dim) or (seq_len, input_dim)
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension

            x = self.input_proj(x)

            for mamba, norm in zip(self.mamba_layers, self.norms):
                x_out = mamba(x)
                x = norm(x + self.dropout(x_out))

            logits = self.output_proj(x)
            return logits.squeeze(0) if logits.size(0) == 1 else logits


class MambaSmootherWrapper:
    """sklearn-style wrapper for Mamba Smoother"""

    def __init__(self, n_classes=4, d_model=64, n_layers=2, epochs=50, lr=1e-3,
                 batch_size=8, aux_dim=0, device=None):
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.aux_dim = aux_dim
        self.device = device or get_device()
        self.label_encoder = None
        self.model = None

    def fit(self, y_pred_proba, y_true, groups, aux_features=None):
        """Train Mamba smoother"""
        print(f"\n[*] Training Mamba Smoother (CUDA: {self.device.type == 'cuda'})")

        # Prepare inputs
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
            self.aux_dim = aux_features.shape[1]
        else:
            inputs = y_pred_proba

        # Handle string labels with LabelEncoder
        if y_true.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            print("  [INFO] Detected string labels, using LabelEncoder")
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y_true)
            print(f"  [INFO] Classes: {self.label_encoder.classes_}")

        # Create dataset
        dataset = SequenceDataset(inputs, y_true, groups, self.label_encoder)

        # Initialize model
        input_dim = inputs.shape[1]
        self.model = MambaSmoother(
            n_classes=self.n_classes,
            d_model=self.d_model,
            n_layers=self.n_layers,
            aux_dim=0,
        ).to(self.device)

        # Fix input_proj dimension
        self.model.input_proj = nn.Linear(input_dim, self.d_model).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_samples = 0

            # Random shuffle sequence order
            indices = np.random.permutation(len(dataset))

            for idx in indices:
                seq, targets = dataset[idx]
                seq = seq.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(seq)
                loss = criterion(logits, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * len(targets)
                n_samples += len(targets)

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / n_samples
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        print("  [OK] Training complete")

    def predict(self, y_pred_proba, groups, aux_features=None):
        """Predict with trained model"""
        if aux_features is not None:
            inputs = np.hstack([y_pred_proba, aux_features])
        else:
            inputs = y_pred_proba

        self.model.eval()
        all_preds = []

        unique_groups = np.unique(groups)
        with torch.no_grad():
            for g in unique_groups:
                mask = groups == g
                seq = torch.FloatTensor(inputs[mask]).to(self.device)
                logits = self.model(seq)
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
    if not MAMBA_AVAILABLE:
        print("\n[!] Mamba-SSM not available")
        print("[!] Please install: pip install mamba-ssm causal-conv1d>=1.1.0")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Train Mamba Smoother (CUDA accelerated)')
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment ID')
    parser.add_argument('--d_model', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--aux_features', type=str, default='enmo',
                       choices=['none', 'enmo', 'full'], help='Auxiliary features')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='./prepared_data')
    parser.add_argument('--output_dir', type=str, default='./models')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Training Mamba Smoother: {args.exp_id} (CUDA accelerated)")
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
        print(f"\n[3/4] Training Mamba Smoother...")
        mamba = MambaSmootherWrapper(
            n_classes=4,
            d_model=args.d_model,
            n_layers=args.n_layers,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

        mamba.fit(y_train_proba, y_train, P_train, aux_train)

        # Evaluation
        print("\n[4/4] Evaluating model...")
        y_pred = mamba.predict(y_test_proba, P_test, aux_test)

        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        print("\n" + "="*70)
        print(f"  {args.exp_id} Test Results")
        print("="*70)
        print(classification_report(y_test, y_pred))
        print(f"\nMacro F1: {f1_macro:.4f}")
        print(f"Per-class F1: {f1_per_class}")

        # Save model
        model_path = output_dir / f'mamba_{args.exp_id}.pkl'
        joblib.dump(mamba, model_path)
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
