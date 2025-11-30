#!/usr/bin/env python
"""
Phase 3: Evaluation
===================
- Generate PR curves comparison
- Generate calibration curves
- Hard-negative evaluation (walking vs chores)
- Export best model
- Measure inference speed
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

plt.style.use("seaborn-v0_8-whitegrid")


def plot_pr_curves(results: Dict[str, Dict], 
                   y_test: np.ndarray,
                   save_path: Path,
                   logger: logging.Logger):
    """
    Plot PR curves for all models on same figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, res in results.items():
        y_proba = res.get('y_proba')
        if y_proba is None:
            continue
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = res['pr_auc']
        ax.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})", linewidth=2)
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves Comparison", fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info(f"  Saved PR curve comparison: {save_path}")


def plot_calibration_curves(results: Dict[str, Dict],
                            y_test: np.ndarray,
                            save_path: Path,
                            logger: logging.Logger):
    """
    Plot calibration curves for all models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    for name, res in results.items():
        y_proba = res.get('y_proba')
        if y_proba is None:
            continue
        
        try:
            frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')
            ax.plot(mean_pred, frac_pos, marker='o', label=name, linewidth=2, markersize=6)
        except Exception as e:
            logger.warning(f"  Could not compute calibration for {name}: {e}")
    
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves (Reliability Diagram)", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info(f"  Saved calibration curves: {save_path}")


def plot_confusion_matrices(results: Dict[str, Dict],
                            save_dir: Path,
                            logger: logging.Logger):
    """
    Plot confusion matrix for each model.
    """
    for name, res in results.items():
        cm = res.get('confusion_matrix')
        if cm is None:
            continue
        
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(5, 4))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Walking'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='.2f')
        ax.set_title(f'{name}\n(Row-normalized)')
        
        fig.tight_layout()
        save_path = save_dir / f"confusion_matrix_{name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"  Saved confusion matrix: {save_path}")


def evaluate_hard_negatives(best_model,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            labels_test: np.ndarray,
                            logger: logging.Logger) -> Dict[str, float]:
    """
    Evaluate on hard-negative subset (walking vs household-chores only).
    """
    # Filter to only walking and household-chores
    hard_mask = np.isin(labels_test, ['walking', 'household-chores'])
    
    if hard_mask.sum() == 0:
        logger.warning("  No hard-negative samples found")
        return {}
    
    X_hard = X_test[hard_mask]
    y_hard = y_test[hard_mask]
    
    logger.info(f"  Hard-negative subset: {len(X_hard)} samples")
    logger.info(f"    Walking: {(y_hard == 1).sum()}")
    logger.info(f"    Household-chores: {(y_hard == 0).sum()}")
    
    # Predict
    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_hard)[:, 1]
    elif hasattr(best_model, 'decision_function'):
        scores = best_model.decision_function(X_hard)
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_proba = best_model.predict(X_hard).astype(float)
    
    pr_auc = average_precision_score(y_hard, y_proba)
    precision, recall, _ = precision_recall_curve(y_hard, y_proba)
    prec_at_recall_70 = precision[recall >= 0.70].max() if (recall >= 0.70).any() else 0
    
    logger.info(f"  Hard-negative PR-AUC: {pr_auc:.4f}")
    logger.info(f"  Hard-negative Prec@Recall>=0.70: {prec_at_recall_70:.4f}")
    
    return {
        'hard_pr_auc': pr_auc,
        'hard_precision_at_recall_70': prec_at_recall_70,
    }


def measure_inference_speed(model,
                            X_sample: np.ndarray,
                            n_runs: int = 5,
                            logger: logging.Logger = None) -> float:
    """
    Measure inference speed (seconds per 1000 windows).
    """
    # Warm up
    _ = model.predict(X_sample[:100])
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    per_1000 = avg_time / len(X_sample) * 1000
    
    if logger:
        logger.info(f"  Inference speed: {per_1000:.4f} s per 1000 windows")
    
    return per_1000


def run_evaluation(args, logger: logging.Logger) -> Dict[str, Any]:
    """
    Main evaluation function.
    """
    features_dir = args.artifacts_dir / "features"
    models_dir = args.artifacts_dir / "models"
    plots_dir = args.artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # Load training results
    # =========================================================================
    logger.info("Loading training results...")
    
    summary_path = args.artifacts_dir / "training_summary.pkl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary not found: {summary_path}")
    
    training_results = joblib.load(summary_path)
    logger.info(f"  Found {len(training_results)} trained models")
    
    # Load test data for plots
    y_test = np.load(features_dir / "Y_binary.npy")
    split = np.load(features_dir / "split_indices.npz")
    test_idx = split['test_idx']
    
    if args.max_samples:
        test_idx = test_idx[test_idx < args.max_samples]
        y_test = y_test[:args.max_samples]
    
    y_test = y_test[test_idx]
    
    # Reload full results with probabilities for each model
    full_results = {}
    for model_name in training_results.keys():
        model_path = models_dir / f"{model_name}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            
            # Determine feature type
            if "handcraft" in model_name:
                X = np.load(features_dir / "handcraft.npy")
            elif "minirocket" in model_name:
                X = np.load(features_dir / "minirocket.npy")
            else:
                continue
            
            if args.max_samples:
                X = X[:args.max_samples]
            
            X_test = X[test_idx]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
                y_proba = 1 / (1 + np.exp(-scores))
            else:
                y_proba = model.predict(X_test).astype(float)
            
            full_results[model_name] = {
                **training_results[model_name],
                'y_proba': y_proba,
                'model': model,
                'X_test': X_test,
            }
    
    # =========================================================================
    # Plot PR curves
    # =========================================================================
    logger.info("Generating PR curve comparison...")
    plot_pr_curves(full_results, y_test, plots_dir / "pr_curves_comparison.png", logger)
    
    # =========================================================================
    # Plot calibration curves
    # =========================================================================
    logger.info("Generating calibration curves...")
    plot_calibration_curves(full_results, y_test, plots_dir / "calibration_curves.png", logger)
    
    # =========================================================================
    # Plot confusion matrices
    # =========================================================================
    logger.info("Generating confusion matrices...")
    plot_confusion_matrices(training_results, plots_dir, logger)
    
    # =========================================================================
    # Find best model
    # =========================================================================
    logger.info("Selecting best model...")
    
    best_name = max(training_results.keys(), 
                    key=lambda k: training_results[k]['pr_auc'])
    best_result = full_results[best_name]
    best_model = best_result['model']
    
    logger.info(f"  Best model: {best_name}")
    logger.info(f"  Best PR-AUC: {training_results[best_name]['pr_auc']:.4f}")
    
    results['best_model'] = best_name
    results['best_pr_auc'] = training_results[best_name]['pr_auc']
    
    # =========================================================================
    # Hard-negative evaluation
    # =========================================================================
    logger.info("Evaluating on hard negatives (walking vs chores)...")
    
    try:
        # Load original labels
        labels = np.load(Path(args.prepared_dir) / "Y_WillettsSpecific2018.npy")
        if args.max_samples:
            labels = labels[:args.max_samples]
        labels_test = labels[test_idx]
        
        hard_results = evaluate_hard_negatives(
            best_model, best_result['X_test'], y_test, labels_test, logger
        )
        results.update(hard_results)
    except Exception as e:
        logger.warning(f"  Could not evaluate hard negatives: {e}")
    
    # =========================================================================
    # Inference speed
    # =========================================================================
    logger.info("Measuring inference speed...")
    
    speed = measure_inference_speed(
        best_model, best_result['X_test'][:1000], n_runs=5, logger=logger
    )
    results['inference_speed_per_1000'] = speed
    
    # =========================================================================
    # Export best model
    # =========================================================================
    logger.info("Exporting best model...")
    
    best_model_path = models_dir / "gait_filter_best.pkl"
    joblib.dump(best_model, best_model_path)
    
    model_size_mb = best_model_path.stat().st_size / (1024 * 1024)
    results['model_size_mb'] = model_size_mb
    logger.info(f"  Saved: {best_model_path} ({model_size_mb:.2f} MB)")
    
    # =========================================================================
    # Write final report
    # =========================================================================
    logger.info("Writing final report...")
    
    report_path = Path("experiments/gait_filter") / f"{args.project_id}_final_report.md"
    
    report_lines = [
        f"# {args.project_id} Gait Filter - Final Report",
        "",
        "## Model Comparison",
        "",
        "| Model | PR-AUC | F1 | Prec@Recall>=0.70 |",
        "|-------|--------|-----|-------------------|",
    ]
    
    for name, res in sorted(training_results.items(), 
                            key=lambda x: x[1]['pr_auc'], reverse=True):
        report_lines.append(
            f"| {name} | {res['pr_auc']:.4f} | {res['f1']:.4f} | {res['precision_at_recall_70']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "## Best Model",
        f"- **Model**: {best_name}",
        f"- **PR-AUC**: {results['best_pr_auc']:.4f}",
        f"- **Model Size**: {model_size_mb:.2f} MB",
        f"- **Inference Speed**: {speed:.4f} s per 1000 windows",
        "",
        "## Hard Negatives (Walking vs Chores)",
        f"- **PR-AUC**: {results.get('hard_pr_auc', 'N/A')}",
        f"- **Prec@Recall>=0.70**: {results.get('hard_precision_at_recall_70', 'N/A')}",
        "",
        "## Artifacts",
        f"- PR Curves: `{plots_dir / 'pr_curves_comparison.png'}`",
        f"- Calibration: `{plots_dir / 'calibration_curves.png'}`",
        f"- Best Model: `{best_model_path}`",
    ])
    
    report_path.write_text("\n".join(report_lines))
    logger.info(f"  Report saved: {report_path}")
    
    logger.info("Evaluation complete!")
    return results


if __name__ == "__main__":
    from pathlib import Path
    
    class Args:
        artifacts_dir = Path("artifacts/gait_filter")
        prepared_dir = Path("prepared_data")
        project_id = "GF_test"
        quick_test = True
        max_samples = 5000
        seed = 42
        n_jobs = 4
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    run_evaluation(Args(), logger)
