#!/usr/bin/env python
"""
Phase 2: Classifier Training
============================
Train multiple classifiers on extracted features:
- Hand-crafted + Random Forest
- Hand-crafted + XGBoost
- Hand-crafted + Logistic Regression
- MiniRocket + Ridge Classifier
- MiniRocket + Logistic Regression
- SAX/SFA + MrSQM (symbolic classifier)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, balanced_accuracy_score, confusion_matrix
)


def load_features_and_labels(features_dir: Path, 
                              feature_type: str,
                              max_samples: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features, labels, and split indices.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load features
    if feature_type == "handcraft":
        X = np.load(features_dir / "handcraft.npy")
    elif feature_type == "minirocket":
        X = np.load(features_dir / "minirocket.npy")
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # Load labels
    y = np.load(features_dir / "Y_binary.npy")
    
    # Load split
    split = np.load(features_dir / "split_indices.npz")
    train_idx = split['train_idx']
    test_idx = split['test_idx']
    
    # Apply sample limit
    if max_samples:
        train_idx = train_idx[train_idx < max_samples]
        test_idx = test_idx[test_idx < max_samples]
        X = X[:max_samples]
        y = y[:max_samples]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def downsample_training(X_train: np.ndarray, 
                        y_train: np.ndarray,
                        max_positive: int = 10000,
                        neg_pos_ratio: int = 4,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample training data to handle imbalance.
    """
    rng = np.random.default_rng(seed)
    
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    
    # Limit positive samples
    if len(pos_idx) > max_positive:
        pos_idx = rng.choice(pos_idx, size=max_positive, replace=False)
    
    # Limit negative samples
    target_neg = min(len(neg_idx), len(pos_idx) * neg_pos_ratio)
    neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
    
    selected = np.concatenate([pos_idx, neg_idx])
    selected.sort()
    
    return X_train[selected], y_train[selected]


def train_and_evaluate(X_train: np.ndarray,
                       X_test: np.ndarray,
                       y_train: np.ndarray,
                       y_test: np.ndarray,
                       model_name: str,
                       model,
                       logger: logging.Logger) -> Dict[str, Any]:
    """
    Train a model and compute evaluation metrics.
    """
    logger.info(f"  Training {model_name}...")
    start = time.time()
    
    # Fit
    model.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info(f"    Training time: {train_time:.1f}s")
    
    # Predict
    start = time.time()
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        # Ridge classifier uses decision function
        scores = model.decision_function(X_test)
        # Convert to pseudo-probabilities using sigmoid
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_proba = y_pred.astype(float)
    
    pred_time = time.time() - start
    
    # Compute metrics
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    # Precision at recall >= 0.70
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    prec_at_recall_70 = precision[recall >= 0.70].max() if (recall >= 0.70).any() else 0
    
    results = {
        'model_name': model_name,
        'train_time_s': train_time,
        'pred_time_s': pred_time,
        'pr_auc': pr_auc,
        'f1': f1,
        'balanced_accuracy': bacc,
        'precision_at_recall_70': prec_at_recall_70,
        'confusion_matrix': cm.tolist(),
        'y_proba': y_proba,
        'y_pred': y_pred,
    }
    
    logger.info(f"    PR-AUC: {pr_auc:.4f}")
    logger.info(f"    F1: {f1:.4f}")
    logger.info(f"    Balanced Accuracy: {bacc:.4f}")
    logger.info(f"    Precision@Recall>=0.70: {prec_at_recall_70:.4f}")
    
    return results


def run_training(args, logger: logging.Logger) -> Dict[str, Any]:
    """
    Main training function.
    
    Trains multiple classifiers and saves results.
    """
    features_dir = args.artifacts_dir / "features"
    models_dir = args.artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # =========================================================================
    # Hand-crafted features + classifiers
    # =========================================================================
    logger.info("Loading hand-crafted features...")
    
    try:
        X_train, X_test, y_train, y_test = load_features_and_labels(
            features_dir, "handcraft", max_samples=args.max_samples
        )
        logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"  Train labels: {(y_train==1).sum()} pos, {(y_train==0).sum()} neg")
        logger.info(f"  Test labels: {(y_test==1).sum()} pos, {(y_test==0).sum()} neg")
        
        # Downsample training
        X_train_ds, y_train_ds = downsample_training(
            X_train, y_train, 
            max_positive=10000, neg_pos_ratio=4, seed=args.seed
        )
        logger.info(f"  After downsampling: {X_train_ds.shape}")
        logger.info(f"    Pos: {(y_train_ds==1).sum()}, Neg: {(y_train_ds==0).sum()}")
        
        # Random Forest
        rf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=500, max_depth=20, 
                n_jobs=args.n_jobs, random_state=args.seed
            ))
        ])
        results_rf = train_and_evaluate(
            X_train_ds, X_test, y_train_ds, y_test,
            "handcraft_rf", rf, logger
        )
        all_results['handcraft_rf'] = results_rf
        joblib.dump(rf, models_dir / "handcraft_rf.pkl")
        
        # XGBoost (if available)
        try:
            from xgboost import XGBClassifier
            xgb = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.1,
                    n_jobs=args.n_jobs, random_state=args.seed,
                    use_label_encoder=False, eval_metric='logloss'
                ))
            ])
            results_xgb = train_and_evaluate(
                X_train_ds, X_test, y_train_ds, y_test,
                "handcraft_xgb", xgb, logger
            )
            all_results['handcraft_xgb'] = results_xgb
            joblib.dump(xgb, models_dir / "handcraft_xgb.pkl")
        except ImportError:
            logger.warning("XGBoost not available, skipping")
        
        # Logistic Regression
        lr = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=2000, solver='saga', n_jobs=args.n_jobs,
                random_state=args.seed
            ))
        ])
        results_lr = train_and_evaluate(
            X_train_ds, X_test, y_train_ds, y_test,
            "handcraft_lr", lr, logger
        )
        all_results['handcraft_lr'] = results_lr
        joblib.dump(lr, models_dir / "handcraft_lr.pkl")
        
    except FileNotFoundError as e:
        logger.warning(f"Hand-crafted features not found: {e}")
    
    # =========================================================================
    # MiniRocket features + classifiers
    # =========================================================================
    logger.info("Loading MiniRocket features...")
    
    try:
        X_train, X_test, y_train, y_test = load_features_and_labels(
            features_dir, "minirocket", max_samples=args.max_samples
        )
        logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Downsample training
        X_train_ds, y_train_ds = downsample_training(
            X_train, y_train,
            max_positive=10000, neg_pos_ratio=4, seed=args.seed
        )
        logger.info(f"  After downsampling: {X_train_ds.shape}")
        
        # Ridge Classifier (standard for Rocket)
        ridge = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RidgeClassifier(alpha=1.0))
        ])
        results_ridge = train_and_evaluate(
            X_train_ds, X_test, y_train_ds, y_test,
            "minirocket_ridge", ridge, logger
        )
        all_results['minirocket_ridge'] = results_ridge
        joblib.dump(ridge, models_dir / "minirocket_ridge.pkl")
        
        # Logistic Regression
        lr = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=2000, solver='saga', n_jobs=args.n_jobs,
                random_state=args.seed
            ))
        ])
        results_lr = train_and_evaluate(
            X_train_ds, X_test, y_train_ds, y_test,
            "minirocket_lr", lr, logger
        )
        all_results['minirocket_lr'] = results_lr
        joblib.dump(lr, models_dir / "minirocket_lr.pkl")
        
    except FileNotFoundError as e:
        logger.warning(f"MiniRocket features not found: {e}")
    
    # =========================================================================
    # SAX + MrSQM
    # =========================================================================
    logger.info("Loading SAX features for MrSQM...")
    
    try:
        from mrsqm import MrSQMClassifier
        
        sax_path = features_dir / "sax.pkl"
        if sax_path.exists():
            sax_words = joblib.load(sax_path)
            
            # Load split indices
            split = np.load(features_dir / "split_indices.npz")
            train_idx = split['train_idx']
            test_idx = split['test_idx']
            
            y = np.load(features_dir / "Y_binary.npy")
            
            if args.max_samples:
                train_idx = train_idx[train_idx < args.max_samples]
                test_idx = test_idx[test_idx < args.max_samples]
                y = y[:args.max_samples]
                sax_words = sax_words[:args.max_samples]
            
            # Convert to numpy array for indexing
            sax_array = np.array(sax_words)
            X_train_sax = sax_array[train_idx]
            X_test_sax = sax_array[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            logger.info(f"  SAX Train: {len(X_train_sax)}, Test: {len(X_test_sax)}")
            
            # Downsample training
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            
            rng = np.random.default_rng(args.seed)
            max_pos = min(len(pos_idx), 10000)
            if len(pos_idx) > max_pos:
                pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)
            target_neg = min(len(neg_idx), len(pos_idx) * 4)
            neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
            
            ds_idx = np.concatenate([pos_idx, neg_idx])
            ds_idx.sort()
            
            X_train_ds = X_train_sax[ds_idx].tolist()  # MrSQM expects list of strings
            y_train_ds = y_train[ds_idx]
            
            logger.info(f"  After downsampling: {len(X_train_ds)} samples")
            
            # Train MrSQM
            logger.info("  Training MrSQM on SAX features...")
            mrsqm = MrSQMClassifier(
                nsax=1,           # 1 SAX representation (we already have it)
                nsfa=0,           # No SFA (separate)
                random_state=args.seed
            )
            
            start = time.time()
            mrsqm.fit(X_train_ds, y_train_ds)
            train_time = time.time() - start
            logger.info(f"    Training time: {train_time:.1f}s")
            
            # Predict
            start = time.time()
            y_pred = mrsqm.predict(X_test_sax.tolist())
            y_proba = mrsqm.predict_proba(X_test_sax.tolist())[:, 1]
            pred_time = time.time() - start
            
            # Metrics
            pr_auc = average_precision_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            prec_at_recall_70 = precision[recall >= 0.70].max() if (recall >= 0.70).any() else 0
            
            results_mrsqm_sax = {
                'model_name': 'sax_mrsqm',
                'train_time_s': train_time,
                'pred_time_s': pred_time,
                'pr_auc': pr_auc,
                'f1': f1,
                'balanced_accuracy': bacc,
                'precision_at_recall_70': prec_at_recall_70,
                'confusion_matrix': cm.tolist(),
                'y_proba': y_proba,
                'y_pred': y_pred,
            }
            
            logger.info(f"    PR-AUC: {pr_auc:.4f}")
            logger.info(f"    F1: {f1:.4f}")
            logger.info(f"    Precision@Recall>=0.70: {prec_at_recall_70:.4f}")
            
            all_results['sax_mrsqm'] = results_mrsqm_sax
            joblib.dump(mrsqm, models_dir / "sax_mrsqm.pkl")
        else:
            logger.warning("  SAX features not found, skipping MrSQM")
            
    except ImportError:
        logger.warning("MrSQM not installed. Install with: pip install mrsqm")
    except Exception as e:
        logger.warning(f"MrSQM training failed: {e}")
    
    # =========================================================================
    # SFA + MrSQM
    # =========================================================================
    logger.info("Loading SFA features for MrSQM...")
    
    try:
        from mrsqm import MrSQMClassifier
        
        sfa_path = features_dir / "sfa.pkl"
        if sfa_path.exists():
            sfa_words = joblib.load(sfa_path)
            
            # Load split indices
            split = np.load(features_dir / "split_indices.npz")
            train_idx = split['train_idx']
            test_idx = split['test_idx']
            
            y = np.load(features_dir / "Y_binary.npy")
            
            if args.max_samples:
                train_idx = train_idx[train_idx < args.max_samples]
                test_idx = test_idx[test_idx < args.max_samples]
                y = y[:args.max_samples]
                sfa_words = sfa_words[:args.max_samples]
            
            sfa_array = np.array(sfa_words)
            X_train_sfa = sfa_array[train_idx]
            X_test_sfa = sfa_array[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            logger.info(f"  SFA Train: {len(X_train_sfa)}, Test: {len(X_test_sfa)}")
            
            # Downsample (reuse indices from SAX if available)
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            
            rng = np.random.default_rng(args.seed)
            max_pos = min(len(pos_idx), 10000)
            if len(pos_idx) > max_pos:
                pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)
            target_neg = min(len(neg_idx), len(pos_idx) * 4)
            neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
            
            ds_idx = np.concatenate([pos_idx, neg_idx])
            ds_idx.sort()
            
            X_train_ds = X_train_sfa[ds_idx].tolist()
            y_train_ds = y_train[ds_idx]
            
            logger.info(f"  After downsampling: {len(X_train_ds)} samples")
            
            # Train MrSQM
            logger.info("  Training MrSQM on SFA features...")
            mrsqm_sfa = MrSQMClassifier(
                nsax=0,           # No SAX
                nsfa=1,           # 1 SFA representation
                random_state=args.seed
            )
            
            start = time.time()
            mrsqm_sfa.fit(X_train_ds, y_train_ds)
            train_time = time.time() - start
            logger.info(f"    Training time: {train_time:.1f}s")
            
            # Predict
            start = time.time()
            y_pred = mrsqm_sfa.predict(X_test_sfa.tolist())
            y_proba = mrsqm_sfa.predict_proba(X_test_sfa.tolist())[:, 1]
            pred_time = time.time() - start
            
            # Metrics
            pr_auc = average_precision_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            prec_at_recall_70 = precision[recall >= 0.70].max() if (recall >= 0.70).any() else 0
            
            results_mrsqm_sfa = {
                'model_name': 'sfa_mrsqm',
                'train_time_s': train_time,
                'pred_time_s': pred_time,
                'pr_auc': pr_auc,
                'f1': f1,
                'balanced_accuracy': bacc,
                'precision_at_recall_70': prec_at_recall_70,
                'confusion_matrix': cm.tolist(),
                'y_proba': y_proba,
                'y_pred': y_pred,
            }
            
            logger.info(f"    PR-AUC: {pr_auc:.4f}")
            logger.info(f"    F1: {f1:.4f}")
            logger.info(f"    Precision@Recall>=0.70: {prec_at_recall_70:.4f}")
            
            all_results['sfa_mrsqm'] = results_mrsqm_sfa
            joblib.dump(mrsqm_sfa, models_dir / "sfa_mrsqm.pkl")
        else:
            logger.warning("  SFA features not found, skipping MrSQM")
            
    except ImportError:
        logger.warning("MrSQM not installed for SFA. Install with: pip install mrsqm")
    except Exception as e:
        logger.warning(f"MrSQM SFA training failed: {e}")
    
    # =========================================================================
    # Save summary
    # =========================================================================
    logger.info("Saving training summary...")
    
    # Remove large arrays for summary
    summary = {}
    for name, res in all_results.items():
        summary[name] = {k: v for k, v in res.items() 
                        if k not in ['y_proba', 'y_pred']}
    
    joblib.dump(summary, args.artifacts_dir / "training_summary.pkl")
    
    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Model':<25} {'PR-AUC':>10} {'F1':>10} {'Prec@R70':>10}")
    logger.info("-" * 60)
    for name, res in summary.items():
        logger.info(f"{name:<25} {res['pr_auc']:>10.4f} {res['f1']:>10.4f} {res['precision_at_recall_70']:>10.4f}")
    
    logger.info("Training complete!")
    return all_results


if __name__ == "__main__":
    from pathlib import Path
    
    class Args:
        artifacts_dir = Path("artifacts/gait_filter")
        quick_test = True
        max_samples = 5000
        seed = 42
        n_jobs = 4
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    run_training(Args(), logger)
