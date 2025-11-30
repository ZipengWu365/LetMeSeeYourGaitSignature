#!/usr/bin/env python
"""
Phase 0: Preprocessing
======================
- Load raw accelerometer data X.npy
- Compute ENMO (Euclidean Norm Minus One) - standard metric for wearable accelerometry
- Create binary walking labels
- Create group-based train/test split

ENMO Definition:
    ENMO = max(||xyz|| - 1g, 0)
         = max(√(x² + y² + z²) - 1, 0)
    
    Where 1g ≈ 1000 mg (milligravity) in normalized units.
    ENMO clips negative values to 0 (when device is stationary, ||xyz|| ≈ 1g).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy.ndimage import median_filter


def compute_enmo(X: np.ndarray, 
                 batch_size: int = 10000,
                 clip_negative: bool = True,
                 apply_median_filter: bool = True,
                 logger: logging.Logger = None) -> np.ndarray:
    """
    Compute ENMO (Euclidean Norm Minus One) from raw 3-axis accelerometer.
    
    ENMO is the standard metric for physical activity intensity from wearables:
        ENMO = max(√(x² + y² + z²) - 1, 0)
    
    Processing steps:
    1. Euclidean norm: ||xyz|| = √(x² + y² + z²)
    2. Subtract 1g (gravity): ||xyz|| - 1
    3. (Optional) Clip negative to 0: max(..., 0)
    4. (Optional) Median filter for noise reduction
    
    Args:
        X: Raw accelerometer data in g units, shape (N, 1000, 3)
        batch_size: Process in batches to manage memory
        clip_negative: If True, ENMO = max(norm-1, 0); if False, allow negative
        apply_median_filter: Apply median filter for noise reduction
        logger: Logger instance
    
    Returns:
        enmo: ENMO values, shape (N, 1000), in g units
    
    Note:
        - Input X should be in g units (1g = 9.8 m/s²)
        - Output ENMO is in g units (typical walking: 0.1-0.3g, running: 0.5-1.0g)
        - For CAPTURE-24, data is already in g units
    """
    n_samples = X.shape[0]
    enmo = np.zeros((n_samples, X.shape[1]), dtype=np.float32)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        if logger:
            logger.info(f"  Computing ENMO: samples {start:,} - {end:,} / {n_samples:,}")
        
        # Load batch (handles mmap)
        batch = np.array(X[start:end])  # shape (batch, 1000, 3)
        
        # Step 1: Euclidean norm ||xyz||
        v = np.linalg.norm(batch, axis=2)  # shape (batch, 1000)
        
        # Step 2: Subtract 1g (gravity component)
        v = v - 1.0
        
        # Step 3: Clip negative values (true ENMO definition)
        if clip_negative:
            v = np.maximum(v, 0.0)
        else:
            # Alternative: allow negative (matches benchmark features.py)
            # and clip extreme values
            v = np.clip(v, -2.0, 2.0)
        
        # Step 4: Optional median filter for noise reduction
        if apply_median_filter:
            for i in range(v.shape[0]):
                v[i] = median_filter(v[i], size=5, mode='nearest')
        
        enmo[start:end] = v.astype(np.float32)
    
    return enmo


def compute_vector_magnitude(X: np.ndarray, 
                              batch_size: int = 10000,
                              logger: logging.Logger = None) -> np.ndarray:
    """
    Compute preprocessed vector magnitude (legacy function, calls compute_enmo).
    
    This version matches benchmark features.py behavior:
    - Allows negative values (clip_negative=False)
    - Applies median filter
    - Clips to [-2, 2]
    """
    return compute_enmo(
        X, 
        batch_size=batch_size,
        clip_negative=False,  # Match benchmark: allow negative
        apply_median_filter=True,
        logger=logger
    )


def create_binary_labels(labels: np.ndarray,
                         positive_classes: set = None,
                         exclude_classes: set = None) -> tuple:
    """
    Create binary walking labels.
    
    Args:
        labels: Original string labels
        positive_classes: Classes to label as 1 (walking)
        exclude_classes: Classes to exclude from dataset
    
    Returns:
        y_binary: Binary labels (0 or 1)
        mask: Boolean mask of included samples
    """
    if positive_classes is None:
        positive_classes = {'walking'}
    if exclude_classes is None:
        exclude_classes = {'mixed-activity'}
    
    labels = labels.astype(str)
    
    # Create exclusion mask
    mask = ~np.isin(labels, list(exclude_classes))
    
    # Create binary labels
    y_binary = np.isin(labels, list(positive_classes)).astype(np.int8)
    
    return y_binary, mask


def create_group_split(participants: np.ndarray,
                       train_ratio: float = 0.8,
                       seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Create participant-based train/test split.
    
    Ensures no participant appears in both train and test sets.
    
    Returns:
        Dictionary with 'train_idx' and 'test_idx' arrays
    """
    rng = np.random.default_rng(seed)
    
    unique_participants = np.unique(participants)
    n_train = int(len(unique_participants) * train_ratio)
    
    rng.shuffle(unique_participants)
    train_participants = set(unique_participants[:n_train])
    test_participants = set(unique_participants[n_train:])
    
    train_mask = np.isin(participants, list(train_participants))
    test_mask = np.isin(participants, list(test_participants))
    
    return {
        'train_idx': np.where(train_mask)[0],
        'test_idx': np.where(test_mask)[0],
        'train_participants': list(train_participants),
        'test_participants': list(test_participants),
    }


def run_preprocessing(args, logger: logging.Logger) -> Dict[str, Any]:
    """
    Main preprocessing function.
    
    Outputs:
        - artifacts/gait_filter/features/V.npy: Vector magnitude (N, 1000)
        - artifacts/gait_filter/features/Y_binary.npy: Binary labels
        - artifacts/gait_filter/features/split_indices.npz: Train/test indices
    """
    import joblib
    
    prepared_dir = Path(args.prepared_dir)
    features_dir = args.artifacts_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # Step 1: Load raw data
    # =========================================================================
    logger.info("Step 1: Loading raw data...")
    
    X = np.load(prepared_dir / "X.npy", mmap_mode='r')
    P = np.load(prepared_dir / "P.npy")
    Y_willetts = np.load(prepared_dir / "Y_WillettsSpecific2018.npy")
    
    n_samples = X.shape[0]
    logger.info(f"  Loaded X: {X.shape} ({X.dtype})")
    logger.info(f"  Loaded P: {P.shape}, unique participants: {len(np.unique(P))}")
    logger.info(f"  Loaded Y: {Y_willetts.shape}, unique labels: {len(np.unique(Y_willetts))}")
    
    # Apply sample limit for quick test
    if args.max_samples and args.max_samples < n_samples:
        logger.info(f"  Quick test: limiting to {args.max_samples} samples")
        indices = np.random.default_rng(args.seed).choice(
            n_samples, size=args.max_samples, replace=False
        )
        indices.sort()
        X = X[indices]
        P = P[indices]
        Y_willetts = Y_willetts[indices]
        n_samples = len(indices)
    
    results['n_samples'] = n_samples
    
    # =========================================================================
    # Step 2: Compute vector magnitude
    # =========================================================================
    logger.info("Step 2: Computing vector magnitude...")
    
    V_path = features_dir / "V.npy"
    if V_path.exists() and not args.quick_test:
        logger.info(f"  Loading cached V from {V_path}")
        V = np.load(V_path, mmap_mode='r')
    else:
        V = compute_vector_magnitude(X, batch_size=10000, logger=logger)
        np.save(V_path, V)
        logger.info(f"  Saved V to {V_path}")
    
    logger.info(f"  V shape: {V.shape}, dtype: {V.dtype}")
    logger.info(f"  V stats: min={V.min():.3f}, max={V.max():.3f}, mean={V.mean():.3f}")
    
    # =========================================================================
    # Step 3: Create binary labels
    # =========================================================================
    logger.info("Step 3: Creating binary labels...")
    
    y_binary, mask = create_binary_labels(
        Y_willetts,
        positive_classes={'walking'},
        exclude_classes={'mixed-activity'}
    )
    
    n_positive = (y_binary == 1).sum()
    n_negative = (y_binary == 0).sum()
    n_excluded = (~mask).sum()
    
    logger.info(f"  Walking (positive): {n_positive:,} ({100*n_positive/n_samples:.1f}%)")
    logger.info(f"  Other (negative): {n_negative:,} ({100*n_negative/n_samples:.1f}%)")
    logger.info(f"  Excluded (mixed-activity): {n_excluded:,}")
    
    np.save(features_dir / "Y_binary.npy", y_binary)
    np.save(features_dir / "mask.npy", mask)
    
    results['n_positive'] = int(n_positive)
    results['n_negative'] = int(n_negative)
    
    # =========================================================================
    # Step 4: Create train/test split
    # =========================================================================
    logger.info("Step 4: Creating group-based train/test split...")
    
    split = create_group_split(P, train_ratio=0.8, seed=args.seed)
    
    logger.info(f"  Train samples: {len(split['train_idx']):,}")
    logger.info(f"  Test samples: {len(split['test_idx']):,}")
    logger.info(f"  Train participants: {len(split['train_participants'])}")
    logger.info(f"  Test participants: {len(split['test_participants'])}")
    
    np.savez(
        features_dir / "split_indices.npz",
        train_idx=split['train_idx'],
        test_idx=split['test_idx']
    )
    joblib.dump({
        'train_participants': split['train_participants'],
        'test_participants': split['test_participants']
    }, features_dir / "split_metadata.pkl")
    
    results['n_train'] = len(split['train_idx'])
    results['n_test'] = len(split['test_idx'])
    
    logger.info("Preprocessing complete!")
    return results


if __name__ == "__main__":
    # Standalone test
    import argparse
    
    class Args:
        prepared_dir = Path("prepared_data")
        artifacts_dir = Path("artifacts/gait_filter")
        quick_test = True
        max_samples = 5000
        seed = 42
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    run_preprocessing(Args(), logger)
