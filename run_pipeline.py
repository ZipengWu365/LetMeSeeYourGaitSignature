#!/usr/bin/env python
"""
CAPTURE-24 Gait Filter Pipeline - One-click Execution Entry Point
=================================================================
This script orchestrates the complete pipeline:
  Phase 0: Preprocessing (vector magnitude, binary labels, train/test split)
  Phase 1: Feature extraction (Hand-crafted, MiniRocket, SAX, SFA)
  Phase 2: Classifier training (RF, XGBoost, LR, Ridge, MrSQM)
  Phase 3: Evaluation and model export

Usage:
  python run_pipeline.py --project-id GF002 --phases all
  python run_pipeline.py --project-id GF002 --phases preprocess,extract
  python run_pipeline.py --project-id GF002 --quick-test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def setup_logging(log_dir: Path, project_id: str) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{project_id}_{timestamp}_execution.log"
    
    logger = logging.getLogger("gait_filter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
    logger.addHandler(ch)
    
    logger.info(f"Log file: {log_file}")
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAPTURE-24 Gait Filter Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project-id", type=str, default="GF002",
                        help="Project identifier for logging and artifacts")
    parser.add_argument("--phases", type=str, default="all",
                        help="Comma-separated phases: preprocess,extract,train,evaluate,all")
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared_data"),
                        help="Directory containing prepared numpy arrays")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/gait_filter"),
                        help="Base directory for outputs")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test mode with small data subset")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to use (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 for all CPUs)")
    parser.add_argument("--skip-minirocket", action="store_true",
                        help="Skip MiniRocket features (saves ~40GB memory)")
    return parser.parse_args()


def run_phase_preprocess(args, logger):
    """Phase 0: Preprocess data - compute vector magnitude, binary labels, splits."""
    # from experiments.gait_filter.preprocess import run_preprocessing
    from preprocess import run_preprocessing
    logger.info("=" * 60)
    logger.info("PHASE 0: PREPROCESSING")
    logger.info("=" * 60)
    return run_preprocessing(args, logger)


def run_phase_extract(args, logger):
    """Phase 1: Extract features using multiple methods."""
    # from experiments.gait_filter.extract_features import run_feature_extraction
    from extract_features import run_feature_extraction
    logger.info("=" * 60)
    logger.info("PHASE 1: FEATURE EXTRACTION")
    logger.info("=" * 60)
    return run_feature_extraction(args, logger)


def run_phase_train(args, logger):
    """Phase 2: Train classifiers on extracted features."""
    # from experiments.gait_filter.train_classifiers import run_training
    from train_classifiers import run_training
    logger.info("=" * 60)
    logger.info("PHASE 2: CLASSIFIER TRAINING")
    logger.info("=" * 60)
    return run_training(args, logger)


def run_phase_evaluate(args, logger):
    """Phase 3: Evaluate models and export best one."""
    # from experiments.gait_filter.evaluate import run_evaluation
    from evaluate import run_evaluation
    logger.info("=" * 60)
    logger.info("PHASE 3: EVALUATION & EXPORT")
    logger.info("=" * 60)
    return run_evaluation(args, logger)


def main():
    args = parse_args()
    
    # Setup directories
    log_dir = Path("experiments/gait_filter/logs")
    args.artifacts_dir = Path(args.artifacts_dir)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir, args.project_id)
    
    logger.info("=" * 60)
    logger.info("CAPTURE-24 GAIT FILTER PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Project ID: {args.project_id}")
    logger.info(f"Phases: {args.phases}")
    logger.info(f"Quick test mode: {args.quick_test}")
    logger.info(f"Prepared data dir: {args.prepared_dir}")
    logger.info(f"Artifacts dir: {args.artifacts_dir}")
    
    if args.quick_test:
        args.max_samples = args.max_samples or 10000
        logger.info(f"Quick test: limiting to {args.max_samples} samples")
    
    # Parse phases
    if args.phases == "all":
        phases = ["preprocess", "extract", "train", "evaluate"]
    else:
        phases = [p.strip() for p in args.phases.split(",")]
    
    start_time = time.time()
    results = {}
    
    try:
        if "preprocess" in phases:
            results["preprocess"] = run_phase_preprocess(args, logger)
        
        if "extract" in phases:
            results["extract"] = run_phase_extract(args, logger)
        
        if "train" in phases:
            results["train"] = run_phase_train(args, logger)
        
        if "evaluate" in phases:
            results["evaluate"] = run_phase_evaluate(args, logger)
        
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise
    
    return results


if __name__ == "__main__":
    main()
