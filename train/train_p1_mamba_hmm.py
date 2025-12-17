#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1: Tiny-Mamba Emission + HMM Decode - Training and Inference Pipeline

Implements the primary model from hmm_based_mamba_design_and_experiments.md

Training process:
1. Load data with participant-level splitting
2. Train Tiny Mamba to predict per-step logits from 32-D features
3. Fit HMM transition matrix from training labels
4. Use Viterbi decoding for inference
5. Evaluate with unified harness
"""

import argparse
import sys
import json
import yaml
import importlib.util
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from local modules
import importlib.util

# Load tiny_mamba_emission
spec = importlib.util.spec_from_file_location("tiny_mamba_emission", project_root / "models" / "tiny_mamba_emission.py")
tiny_mamba_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny_mamba_module)
TinyMambaEmission = tiny_mamba_module.TinyMambaEmission
create_tiny_mamba_emission = tiny_mamba_module.create_tiny_mamba_emission

# Load hmm_decode
spec = importlib.util.spec_from_file_location("hmm_decode", project_root / "models" / "hmm_decode.py")
hmm_decode_module = importlib.util.module_from_spec(spec)
sys.modules['hmm'] = __import__('hmm', fromlist=['HMM'])  # Pre-load hmm module
sys.modules['utils'] = __import__('utils', fromlist=['ordered_unique'])
spec.loader.exec_module(hmm_decode_module)
HMMDecoder = hmm_decode_module.HMMDecoder
create_hmm_decoder = hmm_decode_module.create_hmm_decoder

# Load evaluator
spec = importlib.util.spec_from_file_location("evaluate_sequence_labeling", project_root / "evaluation" / "evaluate_sequence_labeling.py")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)
SequenceLabelingEvaluator = eval_module.SequenceLabelingEvaluator
create_evaluator = eval_module.create_evaluator

from utils import ordered_unique


class SequenceDataset(Dataset):
    """
    Sequence dataset for per-participant sequences
    
    Groups samples by participant (group) and creates contiguous sequences
    for Mamba to process temporal patterns.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ):
        """
        Args:
            X: Feature matrix (n_samples, 32)
            y: Labels (n_samples,) - should already be converted to integer indices
            groups: Participant IDs (n_samples,)
        """
        self.sequences = []
        self.targets = []
        
        unique_groups = ordered_unique(groups)
        for g in unique_groups:
            mask = groups == g
            X_g = torch.FloatTensor(X[mask])
            # Ensure y is integer type
            y_g_values = y[mask]
            if y_g_values.dtype.kind in ['U', 'S', 'O']:
                # Should not happen if called correctly, but handle anyway
                raise ValueError(f"Expected integer labels, got {y_g_values.dtype}")
            y_g = torch.LongTensor(y_g_values.astype(np.int64))
            
            self.sequences.append(X_g)
            self.targets.append(y_g)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.sequences[idx], self.targets[idx]


class P1MambaHMMTrainer:
    """
    Trainer for P1: Tiny-Mamba Emission + HMM Decode
    
    Responsibilities:
    - Train Mamba to emit per-step logits
    - Fit HMM transition matrix
    - Early stopping on validation decoded Macro-F1
    - Logging and artifact saving
    
    Args:
        config: Configuration dictionary
        device: Torch device (cpu, cuda, etc.)
        output_dir: Directory for saving artifacts
    """
    
    def __init__(
        self,
        config: dict,
        device: torch.device = None,
        output_dir: Path = None,
    ):
        self.config = config
        self.device = device or torch.device('cpu')
        self.output_dir = Path(output_dir) if output_dir else Path('./artifacts/p1_hmm')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize model and decoder
        self.mamba = create_tiny_mamba_emission(
            input_dim=config.get('input_dim', 32),
            n_classes=config.get('n_classes', 4),
            d_model=config.get('d_model', 16),
            n_layers=config.get('n_layers', 1),
            dropout=config.get('dropout', 0.2),
            device=self.device,
        )
        
        self.decoder = create_hmm_decoder(n_classes=config.get('n_classes', 4))
        self.evaluator = create_evaluator(n_classes=config.get('n_classes', 4))
        
        # Training state
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_raw_f1': [],
            'val_decoded_f1': [],
        }
        self.best_val_decoded_f1 = -np.inf
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Logger
        self.log_file = self.output_dir / 'training.log'
    
    def _log(self, msg: str):
        """Log message to file and stdout"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_val: np.ndarray,
    ) -> dict:
        """
        Train the P1 model
        
        Args:
            X_train, y_train, groups_train: Training data
            X_val, y_val, groups_val: Validation data
        
        Returns:
            training_result: Dictionary with training metrics
        """
        self._log("\n" + "="*70)
        self._log("P1 TRAINING: Tiny-Mamba Emission + HMM Decode")
        self._log("="*70)
        
        # Create label mapping if labels are strings
        label_to_idx = None
        if y_train.dtype.kind in ['U', 'S', 'O']:
            unique_labels = np.unique(y_train)
            label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self._log(f"\nLabel mapping: {label_to_idx}")
            # Convert string labels to indices for training
            y_train_idx = np.array([label_to_idx[label] for label in y_train])
            y_val_idx = np.array([label_to_idx[label] for label in y_val])
        else:
            y_train_idx = y_train
            y_val_idx = y_val
        
        # Fit HMM transition matrix on training data
        self._log("\n[1/3] Fitting HMM transition matrix on training labels...")
        self.decoder.fit_transition(y_train_idx, groups_train)
        
        # Create datasets
        train_dataset = SequenceDataset(X_train, y_train_idx, groups_train)
        val_dataset = SequenceDataset(X_val, y_val_idx, groups_val)
        
        self._log(f"  Training sequences: {len(train_dataset)}")
        self._log(f"  Validation sequences: {len(val_dataset)}")
        
        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.mamba.parameters(),
            lr=self.config.get('lr', 3e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
        )
        criterion = nn.CrossEntropyLoss()
        
        epochs = self.config.get('epochs', 20)
        early_stopping_patience = self.config.get('early_stopping_patience', 5)
        gradient_clip = self.config.get('gradient_clip', 1.0)
        
        self._log(f"\n[2/3] Training Mamba emission model...")
        self._log(f"  Epochs: {epochs}")
        self._log(f"  LR: {self.config.get('lr', 3e-4)}")
        self._log(f"  Weight decay: {self.config.get('weight_decay', 1e-4)}")
        self._log(f"  Early stopping patience: {early_stopping_patience}")
        
        # Training loop
        self.mamba.train()
        for epoch in range(epochs):
            # Training step
            train_loss = self._train_epoch(train_dataset, optimizer, criterion, gradient_clip)
            
            # Validation step
            val_raw_f1, val_decoded_f1 = self._validate_epoch(val_dataset, y_val_idx, groups_val)
            
            # Log metrics
            self.train_history['epoch'].append(epoch)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_raw_f1'].append(val_raw_f1)
            self.train_history['val_decoded_f1'].append(val_decoded_f1)
            
            self._log(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Val Raw F1: {val_raw_f1:.4f} | Val Decoded F1: {val_decoded_f1:.4f}")
            
            # Early stopping
            if val_decoded_f1 > self.best_val_decoded_f1:
                self.best_val_decoded_f1 = val_decoded_f1
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best checkpoint
                self._save_checkpoint('best_model')
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    self._log(f"\n  [EARLY STOPPING] No improvement for {early_stopping_patience} epochs")
                    break
        
        # Load best model
        self._load_checkpoint('best_model')
        self._log(f"\n  [OK] Best model from epoch {self.best_epoch+1}, Val Decoded F1: {self.best_val_decoded_f1:.4f}")
        
        # Save training history
        self._save_training_history()
        
        return {
            'best_epoch': self.best_epoch,
            'best_val_decoded_f1': self.best_val_decoded_f1,
            'total_epochs': epoch + 1,
        }
    
    def _train_epoch(
        self,
        dataset: SequenceDataset,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        gradient_clip: float,
    ) -> float:
        """Single training epoch"""
        self.mamba.train()
        total_loss = 0
        n_samples = 0
        
        # Random shuffle order of sequences
        indices = np.random.permutation(len(dataset))
        
        for idx in indices:
            X_seq, y_seq = dataset[idx]
            X_seq = X_seq.to(self.device)
            y_seq = y_seq.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.mamba(X_seq)  # (seq_len, n_classes)
            loss = criterion(logits, y_seq)
            
            # Backward pass
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.mamba.parameters(), max_norm=gradient_clip)
            optimizer.step()
            
            total_loss += loss.item() * len(y_seq)
            n_samples += len(y_seq)
        
        return total_loss / n_samples if n_samples > 0 else 0
    
    def _validate_epoch(
        self,
        dataset: SequenceDataset,
        y_true_all: np.ndarray,
        groups_all: np.ndarray,
    ) -> tuple:
        """Single validation epoch, returns raw and decoded F1"""
        self.mamba.eval()
        
        y_pred_raw_list = []
        y_proba_all_list = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                X_seq, _ = dataset[idx]
                X_seq = X_seq.to(self.device)
                
                # Get logits and convert to probabilities
                logits = self.mamba(X_seq)  # (seq_len, n_classes)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()  # (seq_len, n_classes)
                
                # Raw prediction (argmax per time step)
                y_pred_raw = np.argmax(proba, axis=-1)  # (seq_len,)
                y_pred_raw_list.append(y_pred_raw)
                y_proba_all_list.append(proba)
        
        # Concatenate all sequences (should match y_true_all length)
        y_pred_raw = np.concatenate(y_pred_raw_list)
        y_proba_all = np.concatenate(y_proba_all_list)
        
        # Decoded prediction (HMM Viterbi per group)
        y_pred_decoded = self.decoder.predict(y_proba_all, groups=groups_all)
        
        # Compute F1 scores
        from sklearn.metrics import f1_score
        raw_f1 = f1_score(y_true_all, y_pred_raw, average='macro', zero_division=0)
        decoded_f1 = f1_score(y_true_all, y_pred_decoded, average='macro', zero_division=0)
        
        return raw_f1, decoded_f1
    
    def _save_checkpoint(self, name: str = 'best_model'):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            'model_state_dict': self.mamba.state_dict(),
            'config': self.config,
        }, checkpoint_dir / f'{name}.pt')
    
    def _load_checkpoint(self, name: str = 'best_model'):
        """Load model checkpoint"""
        checkpoint_path = self.output_dir / 'checkpoints' / f'{name}.pt'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.mamba.load_state_dict(checkpoint['model_state_dict'])
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def predict(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        groups_test: np.ndarray,
    ) -> dict:
        """
        Inference on test set
        
        Args:
            X_test, y_test, groups_test: Test data
        
        Returns:
            predictions: Dictionary with predictions and metrics
        """
        self._log("\n[3/3] Testing on test set...")
        
        # Convert string labels to indices if needed
        label_to_idx = None
        if y_test.dtype.kind in ['U', 'S', 'O']:
            unique_labels = np.unique(y_test)
            label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            y_test_idx = np.array([label_to_idx[label] for label in y_test])
        else:
            y_test_idx = y_test
        
        self.mamba.eval()
        test_dataset = SequenceDataset(X_test, y_test_idx, groups_test)
        
        y_pred_raw_list = []
        y_proba_list = []
        
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                X_seq, _ = test_dataset[idx]
                X_seq = X_seq.to(self.device)
                
                logits = self.mamba(X_seq)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
                
                y_pred_raw_list.append(np.argmax(proba, axis=-1))
                y_proba_list.append(proba)
        
        # Concatenate
        y_pred_raw = np.concatenate(y_pred_raw_list)
        y_proba_all = np.concatenate(y_proba_list)
        
        # Decode with HMM
        y_pred_decoded = self.decoder.predict(y_proba_all, groups=groups_test)
        
        # Evaluate
        results = self.evaluator.evaluate(
            y_test_idx, y_pred_raw, y_pred_decoded, groups_test,
            return_predictions=True
        )
        
        # Log results
        self.evaluator.print_results(results, f"P1 Test Results")
        
        # Save results
        self.evaluator.save_results(results, self.output_dir / 'test_results')
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Train P1: Tiny-Mamba Emission + HMM Decode'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='prepared_data',
        help='Data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='artifacts/p1_hmm',
        help='Output directory for artifacts'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Configuration YAML file (if None, use defaults)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=None,
        help='Mamba hidden dimension (overrides config)'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=None,
        help='Number of Mamba layers (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    print("\n[*] Loading data...")
    data_dir = Path(args.data_dir)
    X_feats = pd.read_pickle(data_dir / 'X_feats.pkl').values
    Y = np.load(data_dir / 'Y_Walmsley2020.npy')
    P = np.load(data_dir / 'P.npy')
    
    print(f"  X shape: {X_feats.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  P shape: {P.shape}")
    print(f"  Unique participants: {len(np.unique(P))}")
    print(f"  Label distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")
    
    # Split data: train/val/test
    print("\n[*] Splitting data (participant-level)...")
    unique_participants = sorted(np.unique(P))
    n_total = len(unique_participants)
    n_train = int(0.7 * n_total)  # 70% train
    n_val = int(0.15 * n_total)   # 15% val
    # n_test = n_total - n_train - n_val  # 15% test
    
    train_participants = unique_participants[:n_train]
    val_participants = unique_participants[n_train:n_train + n_val]
    test_participants = unique_participants[n_train + n_val:]
    
    train_mask = np.isin(P, train_participants)
    val_mask = np.isin(P, val_participants)
    test_mask = np.isin(P, test_participants)
    
    X_train, y_train, P_train = X_feats[train_mask], Y[train_mask], P[train_mask]
    X_val, y_val, P_val = X_feats[val_mask], Y[val_mask], P[val_mask]
    X_test, y_test, P_test = X_feats[test_mask], Y[test_mask], P[test_mask]
    
    print(f"  Train: {X_train.shape[0]:6d} samples, {len(train_participants):3d} participants")
    print(f"  Val:   {X_val.shape[0]:6d} samples, {len(val_participants):3d} participants")
    print(f"  Test:  {X_test.shape[0]:6d} samples, {len(test_participants):3d} participants")
    
    # Load or create config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        # Default config (smoke test)
        config = {
            'input_dim': 32,
            'n_classes': 4,
            'd_model': 16,
            'n_layers': 1,
            'dropout': 0.2,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'epochs': 20,
            'early_stopping_patience': 5,
            'gradient_clip': 1.0,
        }
    
    # Override with command line arguments
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.d_model is not None:
        config['d_model'] = args.d_model
    if args.n_layers is not None:
        config['n_layers'] = args.n_layers
    
    # Save config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Device
    device = torch.device(args.device)
    
    # Train
    trainer = P1MambaHMMTrainer(config, device=device, output_dir=output_dir)
    train_result = trainer.train(X_train, y_train, P_train, X_val, y_val, P_val)
    
    # Test
    test_result = trainer.predict(X_test, y_test, P_test)
    
    print("\n[OK] P1 Training and Testing Complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
