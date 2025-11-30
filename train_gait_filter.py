"""Training pipeline for CAPTURE-24 walking filter using MiniRocket + LogisticRegression.
All comments and logs are written in English as requested.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, precision_recall_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested

# Matplotlib defaults for consistent plots
plt.style.use("seaborn-v0_8-whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniRocket walking filter")
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared_data"),
                        help="Directory containing prepared numpy arrays")
    parser.add_argument("--artifacts-dir", type=Path,
                        default=Path("artifacts") / "gait_filter",
                        help="Base directory for plots and model artifacts")
    parser.add_argument("--project-id", type=str, default="GF001",
                        help="Identifier used for log file naming")
    parser.add_argument("--max-positive", type=int, default=8000,
                        help="Maximum number of walking windows to keep for training")
    parser.add_argument("--neg-pos-ratio", type=int, default=4,
                        help="Negative to positive ratio after downsampling")
    parser.add_argument("--train-groups", type=int, default=120,
                        help="Approximate number of participants to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Batch size (number of windows) for MiniRocket transforms")
    return parser.parse_args()


def load_prepared_arrays(prepared_dir: Path) -> Dict[str, np.ndarray]:
    arrays = {}
    arrays['X'] = np.load(prepared_dir / 'X.npy', mmap_mode='r')
    arrays['participants'] = np.load(prepared_dir / 'P.npy')
    arrays['timestamps'] = np.load(prepared_dir / 'T.npy')
    arrays['labels_willetts'] = np.load(prepared_dir / 'Y_WillettsSpecific2018.npy')
    arrays['labels_walmsley'] = np.load(prepared_dir / 'Y_Walmsley2020.npy')
    return arrays


def build_binary_mapping(labels: np.ndarray,
                         participant_ids: np.ndarray,
                         positive_labels: Iterable[str],
                         exclude_labels: Iterable[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = labels.astype(str)
    mask = ~np.isin(labels, list(exclude_labels))
    filtered_labels = labels[mask]
    filtered_participants = participant_ids[mask]
    binary = np.isin(filtered_labels, list(positive_labels)).astype(int)
    return mask, binary, filtered_participants


def make_group_split(participants: np.ndarray,
                      train_groups: int,
                      seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(participants)
    train_ratio = min(train_groups / len(unique_groups), 0.95)
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    indices = np.arange(len(participants))
    train_idx, test_idx = next(splitter.split(indices, groups=participants))
    train_mask = np.zeros(len(participants), dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask
    return train_mask, test_mask


def downsample_training_indices(y_binary: np.ndarray,
                                train_mask: np.ndarray,
                                max_positive: int,
                                neg_pos_ratio: int,
                                seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive_idx = np.where(train_mask & (y_binary == 1))[0]
    negative_idx = np.where(train_mask & (y_binary == 0))[0]

    if positive_idx.size == 0:
        raise ValueError("No walking windows found in the selected training groups.")

    if positive_idx.size > max_positive:
        positive_idx = rng.choice(positive_idx, size=max_positive, replace=False)

    target_negative = min(negative_idx.size, positive_idx.size * neg_pos_ratio)
    negative_idx = rng.choice(negative_idx, size=target_negative, replace=False)

    selected = np.concatenate([positive_idx, negative_idx])
    selected.sort()
    return selected


def _to_nested_view(X: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    subset = X[indices]
    subset = np.transpose(subset, (0, 2, 1))  # convert to (n_cases, n_channels, n_time)
    nested = from_3d_numpy_to_nested(subset)
    return nested


def batched_predict(pipeline: Pipeline,
                    X: np.ndarray,
                    indices: np.ndarray,
                    batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    preds: List[int] = []
    probas: List[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        nested = _to_nested_view(X, batch_idx)
        probas.append(pipeline.predict_proba(nested)[:, 1])
        preds.append(pipeline.predict(nested))
    return np.concatenate(preds), np.concatenate(probas)


def plot_precision_recall(y_true: np.ndarray,
                          probas: np.ndarray,
                          path: Path) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probas)
    pr_auc = average_precision_score(y_true, probas)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    target_precision = None
    if (recall >= 0.70).any():
        target_precision = precision[recall >= 0.70].max()
    return {"pr_auc": pr_auc, "precision_at_recall_ge_0.70": target_precision or float('nan')}


def plot_calibration(y_true: np.ndarray,
                     probas: np.ndarray,
                     path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, probas, n_bins=10, strategy='uniform')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted value')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Reliability Diagram')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray,
                          preds: np.ndarray,
                          path: Path) -> np.ndarray:
    cm = confusion_matrix(y_true, preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Walking'])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Row-normalized Confusion Matrix')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return cm


def measure_inference_speed(pipeline: Pipeline,
                             X: np.ndarray,
                             indices: np.ndarray,
                             batch_size: int) -> float:
    subset = indices[:1000] if indices.size >= 1000 else indices
    nested = _to_nested_view(X, subset)
    start = time.perf_counter()
    pipeline.predict_proba(nested)
    duration = time.perf_counter() - start
    windows = len(subset)
    return duration / windows * 1000  # seconds per 1000 windows


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    plots_dir = artifacts_dir / 'plots'
    models_dir = artifacts_dir / 'models'
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_prepared_arrays(args.prepared_dir)
    X = arrays['X']
    participants = arrays['participants']
    labels = arrays['labels_willetts']

    positive_labels = {'walking', 'sports'}  # treat sports as potential running
    exclude_labels = {'mixed-activity'}
    mask, y_binary, filtered_participants = build_binary_mapping(labels, participants,
                                                                 positive_labels, exclude_labels)
    filtered_indices = np.where(mask)[0]
    labels_filtered = labels[mask]

    train_mask, test_mask = make_group_split(filtered_participants,
                                             train_groups=args.train_groups,
                                             seed=args.seed)

    train_rel_idx = downsample_training_indices(y_binary,
                                                train_mask,
                                                args.max_positive,
                                                args.neg_pos_ratio,
                                                args.seed)
    test_rel_idx = np.where(test_mask)[0]
    train_indices = filtered_indices[train_rel_idx]
    test_indices = filtered_indices[test_rel_idx]

    y_train = y_binary[train_rel_idx]
    print(f"Training windows: {len(train_indices)}, positives: {(y_train == 1).sum()},"
          f" negatives: {(y_train == 0).sum()}")
    print(f"Test windows (unbalanced): {len(test_indices)}")

    X_train_nested = _to_nested_view(X, train_indices)

    minirocket = MiniRocketMultivariate(random_state=args.seed)
    clf = LogisticRegression(max_iter=2000, solver='saga', n_jobs=-1,
                             class_weight=None)
    pipeline = Pipeline([
        ('minirocket', minirocket),
        ('logreg', clf)
    ])

    print("Fitting MiniRocket + LogisticRegression ...")
    pipeline.fit(X_train_nested, y_train)

    print("Evaluating on held-out participants ...")
    full_preds, full_probas = batched_predict(pipeline,
                                              X,
                                              test_indices,
                                              args.batch_size)
    y_test = y_binary[test_rel_idx]

    metrics = plot_precision_recall(y_test, full_probas,
                                    plots_dir / 'precision_recall_curve.png')
    plot_calibration(y_test, full_probas, plots_dir / 'calibration_curve.png')
    cm = save_confusion_matrix(y_test, full_preds,
                               plots_dir / 'confusion_matrix.png')

    hard_mask = np.isin(labels_filtered[test_rel_idx], ['walking', 'household-chores'])
    hard_indices = test_indices[hard_mask]
    hard_rel = test_rel_idx[hard_mask]
    hard_preds, hard_probas = batched_predict(pipeline,
                                              X,
                                              hard_indices,
                                              args.batch_size)
    hard_y = y_binary[hard_rel]
    hard_pr = plot_precision_recall(hard_y, hard_probas,
                                    plots_dir / 'precision_recall_hard.png')

    model_path = models_dir / 'gait_filter.pkl'
    joblib.dump(pipeline, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    seconds_per_k = measure_inference_speed(pipeline,
                                            X,
                                            test_indices,
                                            args.batch_size)

    log_path = Path('experiments') / 'gait_filter' / f"{args.project_id}_capture24_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_payload = {
        'project_id': args.project_id,
        'train_windows': int(len(train_indices)),
        'test_windows': int(len(test_indices)),
        'train_positive': int((y_train == 1).sum()),
        'train_negative': int((y_train == 0).sum()),
        'test_positive': int((y_test == 1).sum()),
        'test_negative': int((y_test == 0).sum()),
        'precision_recall': metrics,
        'hard_negative_precision_recall': hard_pr,
        'confusion_matrix_row_normalized': cm.tolist(),
        'model_size_mb': model_size_mb,
        'seconds_per_1000_windows': seconds_per_k,
    }

    log_lines = [
        f"# {args.project_id} MiniRocket Walking Filter",
        "",
        "## Dataset",
        f"- Total filtered windows: {len(filtered_indices)}",
        f"- Train participants (approx): {args.train_groups}",
        f"- Train windows used: {len(train_indices)} (pos {log_payload['train_positive']}, neg {log_payload['train_negative']})",
        f"- Test windows: {len(test_indices)} (pos {log_payload['test_positive']}, neg {log_payload['test_negative']})",
        "",
        "## Metrics",
        f"- PR AUC: {metrics['pr_auc']:.3f}",
        f"- Precision at recall ≥ 0.70: {metrics['precision_at_recall_ge_0.70']:.3f}",
        f"- Hard-negative PR AUC: {hard_pr['pr_auc']:.3f}",
        f"- Hard-negative precision at recall ≥ 0.70: {hard_pr['precision_at_recall_ge_0.70']:.3f}",
        "",
        "## Artifacts",
        f"- Confusion matrix: {plots_dir / 'confusion_matrix.png'}",
        f"- PR curve: {plots_dir / 'precision_recall_curve.png'}",
        f"- Calibration: {plots_dir / 'calibration_curve.png'}",
        f"- Hard-negative PR: {plots_dir / 'precision_recall_hard.png'}",
        f"- Model path: {model_path} ({model_size_mb:.2f} MB)",
        f"- Inference time: {seconds_per_k:.4f} s per 1000 windows",
        "",
        "## JSON summary",
        "```json",
        json.dumps(log_payload, indent=2),
        "```",
    ]
    log_path.write_text("\n".join(log_lines))
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
