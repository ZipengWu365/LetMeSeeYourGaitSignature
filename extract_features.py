#!/usr/bin/env python
"""
Phase 1: Feature Extraction
===========================
Extract multiple feature representations from vector magnitude:
- Hand-crafted features (31-dim) - reuse from features.py
- MiniRocket features (~10k-dim)
- SAX symbolic representation
- SFA symbolic representation
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import joblib


def extract_handcraft_features(V: np.ndarray,
                                sample_rate: int = 100,
                                batch_size: int = 10000,
                                n_jobs: int = -1,
                                logger: logging.Logger = None) -> np.ndarray:
    """
    Extract hand-crafted features matching benchmark features.py
    
    Since V is already preprocessed (norm, median filter, detrend, clip),
    we apply the same feature functions.
    """
    import scipy.stats as stats
    import scipy.signal as signal
    import statsmodels.tsa.stattools as stattools
    from joblib import Parallel, delayed
    
    def extract_single(v):
        """Extract features for a single window."""
        feats = []
        
        # Moments (4)
        avg = np.mean(v)
        std = np.std(v)
        if std > 0.01:
            skew = np.nan_to_num(stats.skew(v))
            kurt = np.nan_to_num(stats.kurtosis(v))
        else:
            skew = kurt = 0
        feats.extend([avg, std, skew, kurt])
        
        # Quantiles (5)
        q = np.quantile(v, [0, 0.25, 0.5, 0.75, 1])
        feats.extend(q)
        
        # Autocorrelation (5)
        with np.errstate(divide='ignore', invalid='ignore'):
            acf = np.nan_to_num(stattools.acf(v, nlags=2 * sample_rate))
        
        peaks, _ = signal.find_peaks(acf, prominence=0.1)
        if len(peaks) > 0:
            acf_1st_max = acf[peaks[0]]
            acf_1st_max_loc = peaks[0] / sample_rate
        else:
            acf_1st_max = acf_1st_max_loc = 0.0
        
        valleys, _ = signal.find_peaks(-acf, prominence=0.1)
        if len(valleys) > 0:
            acf_1st_min = acf[valleys[0]]
            acf_1st_min_loc = valleys[0] / sample_rate
        else:
            acf_1st_min = acf_1st_min_loc = 0.0
        
        acf_zeros = np.sum(np.diff(np.signbit(acf)))
        feats.extend([acf_1st_max, acf_1st_max_loc, acf_1st_min, acf_1st_min_loc, acf_zeros])
        
        # Spectral (8)
        freqs, powers = signal.periodogram(v, fs=sample_rate, detrend='constant', scaling='density')
        powers = powers / (len(v) / sample_rate)
        
        pentropy = stats.entropy(powers[powers > 0]) if (powers > 0).any() else 0
        power = np.sum(powers)
        
        peaks_idx, _ = signal.find_peaks(powers)
        peak_powers = powers[peaks_idx]
        peak_freqs = freqs[peaks_idx]
        peak_ranks = np.argsort(peak_powers)[::-1]
        
        f1, f2, f3 = 0, 0, 0
        p1, p2, p3 = 0, 0, 0
        for i, j in enumerate(peak_ranks[:3]):
            if i == 0:
                f1, p1 = peak_freqs[j], peak_powers[j]
            elif i == 1:
                f2, p2 = peak_freqs[j], peak_powers[j]
            elif i == 2:
                f3, p3 = peak_freqs[j], peak_powers[j]
        
        feats.extend([pentropy, power, f1, p1, f2, p2, f3, p3])
        
        # FFT Welch (6)
        _, welch_powers = signal.welch(
            v, fs=sample_rate,
            nperseg=sample_rate,
            noverlap=sample_rate // 2,
            detrend='constant',
            scaling='density',
            average='median'
        )
        fft_feats = welch_powers[:6] if len(welch_powers) >= 6 else list(welch_powers) + [0] * (6 - len(welch_powers))
        feats.extend(fft_feats)
        
        # Peaks (4)
        def butterfilt(x, cutoff, fs, order=4):
            from scipy.signal import butter, sosfiltfilt
            nyq = 0.5 * fs
            sos = butter(order, cutoff / nyq, btype='low', analog=False, output='sos')
            return sosfiltfilt(sos, x)
        
        v_filt = butterfilt(v, 5, sample_rate)
        peaks_idx, peak_props = signal.find_peaks(
            v_filt, distance=int(0.2 * sample_rate), prominence=0.25
        )
        npeaks = len(peaks_idx) / (len(v) / sample_rate)
        if len(peak_props.get('prominences', [])) > 0:
            prom = peak_props['prominences']
            peaks_avg = np.mean(prom)
            peaks_min = np.min(prom)
            peaks_max = np.max(prom)
        else:
            peaks_avg = peaks_min = peaks_max = 0
        feats.extend([npeaks, peaks_avg, peaks_min, peaks_max])
        
        return np.array(feats, dtype=np.float32)
    
    n_samples = V.shape[0]
    
    if logger:
        logger.info(f"  Extracting hand-crafted features for {n_samples:,} samples...")
    
    # Parallel extraction
    features = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(extract_single)(V[i]) for i in range(n_samples)
    )
    
    return np.array(features, dtype=np.float32)


def extract_minirocket_features(V: np.ndarray,
                                 batch_size: int = 5000,
                                 seed: int = 42,
                                 logger: logging.Logger = None) -> np.ndarray:
    """
    Extract MiniRocket features from vector magnitude.
    
    Note: MiniRocket expects shape (n_samples, n_timepoints) for univariate.
    """
    try:
        from sktime.transformations.panel.rocket import MiniRocket
    except ImportError:
        logger.warning("sktime not available, skipping MiniRocket")
        return None
    
    n_samples = V.shape[0]
    
    if logger:
        logger.info(f"  Extracting MiniRocket features for {n_samples:,} samples...")
    
    # MiniRocket for univariate
    minirocket = MiniRocket(random_state=seed)
    
    # Fit on first batch
    if logger:
        logger.info("  Fitting MiniRocket on first batch...")
    
    fit_size = min(batch_size, n_samples)
    V_fit = V[:fit_size].reshape(fit_size, 1, -1)  # (batch, 1, time)
    
    # Convert to pandas for sktime
    import pandas as pd
    from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
    
    V_fit_nested = from_3d_numpy_to_nested(V_fit)
    minirocket.fit(V_fit_nested)
    
    # Transform all data in batches
    n_features = minirocket.transform(V_fit_nested).shape[1]
    features = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        if logger:
            logger.info(f"  Transforming batch {start:,} - {end:,}")
        
        V_batch = V[start:end].reshape(end - start, 1, -1)
        V_batch_nested = from_3d_numpy_to_nested(V_batch)
        features[start:end] = minirocket.transform(V_batch_nested).values.astype(np.float32)
    
    return features


def extract_sax_features(V: np.ndarray,
                          word_length: int = 16,
                          alphabet_size: int = 4,
                          logger: logging.Logger = None) -> List[str]:
    """
    Extract SAX symbolic representation.
    """
    try:
        from pyts.transformation import SymbolicAggregateApproximation
    except ImportError:
        logger.warning("pyts not available, skipping SAX")
        return None
    
    n_samples = V.shape[0]
    
    if logger:
        logger.info(f"  Extracting SAX features (w={word_length}, a={alphabet_size})...")
    
    sax = SymbolicAggregateApproximation(
        n_bins=alphabet_size,
        strategy='quantile',
        alphabet='ordinal'
    )
    
    # SAX returns integer codes, convert to strings
    sax_codes = sax.fit_transform(V)
    
    # Convert to word strings
    sax_words = [''.join(chr(ord('a') + c) for c in row) for row in sax_codes]
    
    return sax_words


def extract_sfa_features(V: np.ndarray,
                          word_length: int = 8,
                          alphabet_size: int = 4,
                          logger: logging.Logger = None) -> List[str]:
    """
    Extract SFA symbolic representation.
    """
    try:
        from pyts.transformation import SymbolicFourierApproximation
    except ImportError:
        logger.warning("pyts not available, skipping SFA")
        return None
    
    n_samples = V.shape[0]
    
    if logger:
        logger.info(f"  Extracting SFA features (w={word_length}, a={alphabet_size})...")
    
    sfa = SymbolicFourierApproximation(
        n_coefs=word_length,
        n_bins=alphabet_size,
        strategy='quantile',
        alphabet='ordinal'
    )
    
    sfa_codes = sfa.fit_transform(V)
    sfa_words = [''.join(chr(ord('a') + c) for c in row) for row in sfa_codes]
    
    return sfa_words


def run_feature_extraction(args, logger: logging.Logger) -> Dict[str, Any]:
    """
    Main feature extraction function.
    
    Outputs:
        - artifacts/gait_filter/features/handcraft.npy
        - artifacts/gait_filter/features/minirocket.npy
        - artifacts/gait_filter/features/sax.pkl
        - artifacts/gait_filter/features/sfa.pkl
    """
    features_dir = args.artifacts_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # Load vector magnitude
    # =========================================================================
    logger.info("Loading vector magnitude V...")
    
    V_path = features_dir / "V.npy"
    if not V_path.exists():
        raise FileNotFoundError(f"V.npy not found. Run preprocessing first: {V_path}")
    
    V = np.load(V_path, mmap_mode='r')
    
    # Apply sample limit for quick test
    if args.max_samples and args.max_samples < len(V):
        logger.info(f"  Quick test: limiting to {args.max_samples} samples")
        V = np.array(V[:args.max_samples])
    else:
        V = np.array(V)  # Load into memory for faster processing
    
    logger.info(f"  V shape: {V.shape}")
    
    # =========================================================================
    # Extract hand-crafted features
    # =========================================================================
    logger.info("Extracting hand-crafted features...")
    
    handcraft_path = features_dir / "handcraft.npy"
    if handcraft_path.exists() and not args.quick_test:
        logger.info(f"  Loading cached from {handcraft_path}")
        handcraft = np.load(handcraft_path)
    else:
        handcraft = extract_handcraft_features(
            V, sample_rate=100, n_jobs=args.n_jobs, logger=logger
        )
        np.save(handcraft_path, handcraft)
        logger.info(f"  Saved to {handcraft_path}")
    
    logger.info(f"  Hand-crafted shape: {handcraft.shape}")
    results['handcraft_shape'] = handcraft.shape
    
    # =========================================================================
    # Extract MiniRocket features
    # =========================================================================
    skip_minirocket = getattr(args, 'skip_minirocket', False)
    
    if skip_minirocket:
        logger.info("Skipping MiniRocket features (--skip-minirocket)")
        minirocket = None
    else:
        logger.info("Extracting MiniRocket features...")
        
        minirocket_path = features_dir / "minirocket.npy"
        if minirocket_path.exists() and not args.quick_test:
            logger.info(f"  Loading cached from {minirocket_path}")
            minirocket = np.load(minirocket_path)
        else:
            minirocket = extract_minirocket_features(
                V, batch_size=5000, seed=args.seed, logger=logger
            )
            if minirocket is not None:
                np.save(minirocket_path, minirocket)
                logger.info(f"  Saved to {minirocket_path}")
        
        if minirocket is not None:
            logger.info(f"  MiniRocket shape: {minirocket.shape}")
            results['minirocket_shape'] = minirocket.shape
    
    # =========================================================================
    # Extract SAX features
    # =========================================================================
    logger.info("Extracting SAX features...")
    
    sax_path = features_dir / "sax.pkl"
    if sax_path.exists() and not args.quick_test:
        logger.info(f"  Loading cached from {sax_path}")
        sax_words = joblib.load(sax_path)
    else:
        sax_words = extract_sax_features(V, word_length=16, alphabet_size=4, logger=logger)
        if sax_words is not None:
            joblib.dump(sax_words, sax_path)
            logger.info(f"  Saved to {sax_path}")
    
    if sax_words is not None:
        logger.info(f"  SAX: {len(sax_words)} words, example: {sax_words[0][:50]}...")
        results['sax_count'] = len(sax_words)
    
    # =========================================================================
    # Extract SFA features
    # =========================================================================
    logger.info("Extracting SFA features...")
    
    sfa_path = features_dir / "sfa.pkl"
    if sfa_path.exists() and not args.quick_test:
        logger.info(f"  Loading cached from {sfa_path}")
        sfa_words = joblib.load(sfa_path)
    else:
        sfa_words = extract_sfa_features(V, word_length=8, alphabet_size=4, logger=logger)
        if sfa_words is not None:
            joblib.dump(sfa_words, sfa_path)
            logger.info(f"  Saved to {sfa_path}")
    
    if sfa_words is not None:
        logger.info(f"  SFA: {len(sfa_words)} words, example: {sfa_words[0]}")
        results['sfa_count'] = len(sfa_words)
    
    logger.info("Feature extraction complete!")
    return results


if __name__ == "__main__":
    import argparse
    
    class Args:
        artifacts_dir = Path("artifacts/gait_filter")
        quick_test = True
        max_samples = 1000
        seed = 42
        n_jobs = 4
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    run_feature_extraction(Args(), logger)
