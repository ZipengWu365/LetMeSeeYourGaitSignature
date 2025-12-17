#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Evaluation Harness for Sequence Labeling Models

Single source of truth for metric computation across all models:
- RF+HMM
- ESN Smoother
- Mamba Smoother
- P1 Tiny-Mamba+HMM

Ensures apples-to-apples comparison with identical metric code paths.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from typing import Dict, Tuple, Optional
from pathlib import Path
import json


class SequenceLabelingEvaluator:
    """
    Unified evaluator for sequence labeling models
    
    Computes metrics with identical code paths for all models:
    - Raw F1 (argmax of probabilities, no temporal decoding)
    - Decoded F1 (after Viterbi or other temporal decoding)
    - Per-class metrics
    - Confusion matrix
    
    Args:
        n_classes: Number of output classes (default: 4)
        class_names: Names of output classes (default: None)
    """
    
    def __init__(
        self,
        n_classes: int = 4,
        class_names: Optional[list] = None,
    ):
        self.n_classes = n_classes
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_raw: np.ndarray,
        y_pred_decoded: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        return_predictions: bool = False,
    ) -> Dict:
        """
        Comprehensive evaluation
        
        Args:
            y_true: True labels (n_samples,)
            y_pred_raw: Raw predictions - argmax of probabilities (n_samples,)
            y_pred_decoded: Decoded predictions after temporal smoothing (n_samples,)
            groups: Group identifiers for per-group analysis (n_samples,)
            return_predictions: Whether to return predictions in results
        
        Returns:
            results: Dictionary with all metrics
        """
        results = {}
        
        # Raw metrics (no temporal decoding)
        results['raw'] = self._compute_metrics(y_true, y_pred_raw, 'raw')
        
        # Decoded metrics (if provided)
        if y_pred_decoded is not None:
            results['decoded'] = self._compute_metrics(y_true, y_pred_decoded, 'decoded')
        
        # Per-group analysis (if provided)
        if groups is not None:
            results['per_group'] = self._compute_per_group_metrics(
                y_true, y_pred_raw, y_pred_decoded, groups
            )
        
        # Store predictions if requested
        if return_predictions:
            results['predictions'] = {
                'y_true': y_true,
                'y_pred_raw': y_pred_raw,
                'y_pred_decoded': y_pred_decoded,
                'groups': groups,
            }
        
        return results
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tag: str = '',
    ) -> Dict:
        """
        Compute all metrics for a single prediction set
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            tag: Tag for metric names
        
        Returns:
            metrics: Dictionary with all computed metrics
        """
        metrics_dict = {}
        
        # Primary metric: Macro F1
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics_dict['macro_f1'] = macro_f1
        
        # Per-class F1
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, f1 in enumerate(per_class_f1):
            metrics_dict[f'f1_class_{i}'] = f1
        
        # Other metrics
        metrics_dict['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics_dict['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics_dict['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.n_classes))
        metrics_dict['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            labels=np.arange(self.n_classes),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics_dict['classification_report'] = report
        
        return metrics_dict
    
    def _compute_per_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred_raw: np.ndarray,
        y_pred_decoded: Optional[np.ndarray],
        groups: np.ndarray,
    ) -> Dict:
        """
        Compute metrics per participant/group
        
        Args:
            y_true: True labels
            y_pred_raw: Raw predictions
            y_pred_decoded: Decoded predictions
            groups: Group identifiers
        
        Returns:
            per_group_metrics: Dictionary with per-group F1 scores
        """
        per_group = {}
        unique_groups = np.unique(groups)
        
        for g in unique_groups:
            mask = groups == g
            y_true_g = y_true[mask]
            y_pred_raw_g = y_pred_raw[mask]
            
            group_metrics = {
                'raw_macro_f1': f1_score(y_true_g, y_pred_raw_g, average='macro', zero_division=0),
                'raw_samples': mask.sum(),
            }
            
            if y_pred_decoded is not None:
                y_pred_decoded_g = y_pred_decoded[mask]
                group_metrics['decoded_macro_f1'] = f1_score(
                    y_true_g, y_pred_decoded_g, average='macro', zero_division=0
                )
            
            per_group[str(g)] = group_metrics
        
        return per_group
    
    def print_results(self, results: Dict, tag: str = ''):
        """
        Pretty-print evaluation results
        
        Args:
            results: Results dictionary from evaluate()
            tag: Optional tag for output
        """
        if tag:
            print(f"\n{'='*70}")
            print(f"  {tag}")
            print(f"{'='*70}")
        
        # Raw metrics
        if 'raw' in results:
            raw = results['raw']
            print(f"\n[RAW] Macro F1: {raw['macro_f1']:.4f}")
            print(f"  Per-class F1: {[f'{v:.4f}' for v in [raw[f'f1_class_{i}'] for i in range(self.n_classes)]]}")
            print(f"  Balanced Accuracy: {raw['balanced_accuracy']:.4f}")
            print(f"  Cohen's Kappa: {raw['cohen_kappa']:.4f}")
        
        # Decoded metrics
        if 'decoded' in results:
            decoded = results['decoded']
            print(f"\n[DECODED] Macro F1: {decoded['macro_f1']:.4f}")
            print(f"  Per-class F1: {[f'{v:.4f}' for v in [decoded[f'f1_class_{i}'] for i in range(self.n_classes)]]}")
            print(f"  Balanced Accuracy: {decoded['balanced_accuracy']:.4f}")
            print(f"  Cohen's Kappa: {decoded['cohen_kappa']:.4f}")
        
        # Classification report
        if 'raw' in results and 'classification_report' in results['raw']:
            print("\n[CLASSIFICATION REPORT]")
            report = results['raw']['classification_report']
            # Print summary
            for key in ['accuracy', 'macro avg', 'weighted avg']:
                if key in report:
                    print(f"  {key}: {report[key]}")
    
    def save_results(self, results: Dict, output_dir: Path):
        """
        Save results to disk
        
        Args:
            results: Results dictionary
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save metrics JSON (exclude non-serializable items)
        metrics_to_save = {}
        for key, val in results.items():
            if key not in ['predictions']:
                metrics_to_save[key] = self._make_serializable(val)
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save predictions if present
        if 'predictions' in results:
            preds = results['predictions']
            predictions_df = pd.DataFrame({
                'y_true': preds['y_true'],
                'y_pred_raw': preds['y_pred_raw'],
                'groups': preds['groups'],
            })
            if preds['y_pred_decoded'] is not None:
                predictions_df['y_pred_decoded'] = preds['y_pred_decoded']
            
            predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
        
        print(f"[OK] Results saved to {output_dir}")
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable types"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def create_evaluator(n_classes: int = 4) -> SequenceLabelingEvaluator:
    """
    Factory function to create evaluator
    
    Args:
        n_classes: Number of output classes
    
    Returns:
        evaluator: Initialized evaluator
    """
    return SequenceLabelingEvaluator(
        n_classes=n_classes,
        class_names=['sleep', 'sedentary', 'tasks-light', 'walking'],
    )
