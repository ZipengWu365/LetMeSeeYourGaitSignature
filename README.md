# CAPTURE-24 Gait Filter Pipeline

Walking recognition from wearable accelerometer data using machine learning.

---

## 📋 Prerequisites

1. **Python 3.9+** with pip
2. **CAPTURE-24 prepared data** in `prepared_data/` directory:
   - `X.npy` - Raw accelerometer data (N, 1000, 3)
   - `Y_anno.npy` or `Y.npy` - Activity annotations
   - `P.npy` - Participant IDs

---

## 🚀 Quick Start

### Option 1: Linux One-Click Run (Recommended)

```bash
# From project ROOT directory (important!)
cd capture24-master

# Make script executable and run
chmod +x experiments/gait_filter/run.sh
./experiments/gait_filter/run.sh --quick-test

# Full run
./experiments/gait_filter/run.sh
```

### Option 2: Windows / Manual Python Run

```bash
# From project ROOT directory (important!)
cd capture24-master

# Install core dependencies
pip install numpy scipy pandas scikit-learn joblib matplotlib statsmodels xgboost tqdm

# Quick test (~20 seconds)
python experiments/gait_filter/run_pipeline.py --project-id TEST --max-samples 500 --skip-minirocket

# Full run (all 934k samples, ~1-2 hours)
python experiments/gait_filter/run_pipeline.py --project-id GF001
```

> ⚠️ **Important**: Always run from the project **root** directory, not from `experiments/gait_filter/`

---

## 📦 Dependencies

### Required (Core)
```bash
pip install numpy scipy pandas scikit-learn joblib matplotlib statsmodels xgboost tqdm
```

### Optional (Advanced Features)
```bash
# MiniRocket features (requires ~40GB RAM for full data)
pip install sktime

# SAX/SFA symbolic features
pip install pyts
```

---

## ⚙️ Command Line Options

```bash
python experiments/gait_filter/run_pipeline.py [OPTIONS]

Options:
  --project-id TEXT      Project ID for logs/outputs [default: GF002]
  --phases TEXT          Phases: all, preprocess, extract, train, evaluate
  --prepared-dir PATH    Path to prepared_data/ [default: prepared_data]
  --artifacts-dir PATH   Output directory [default: artifacts/gait_filter]
  --quick-test           Quick test mode (10k samples)
  --max-samples INT      Limit samples (for testing)
  --skip-minirocket      Skip MiniRocket features (saves ~40GB RAM)
  --seed INT             Random seed [default: 42]
  --n-jobs INT           Parallel jobs [default: -1]
```

### Examples

```bash
# Quick validation
python experiments/gait_filter/run_pipeline.py --project-id TEST --quick-test --skip-minirocket

# Limit samples
python experiments/gait_filter/run_pipeline.py --project-id GF002 --max-samples 100000

# Run specific phases
python experiments/gait_filter/run_pipeline.py --project-id GF002 --phases train,evaluate
```

---

## 📊 Pipeline Phases

| Phase | Description | Time (10k) | Time (934k) |
|-------|-------------|------------|-------------|
| 0. Preprocess | Compute ENMO, binary labels | ~2s | ~3 min |
| 1. Extract | Hand-crafted features (32-dim) | ~5s | ~10 min |
| 2. Train | RF, XGBoost, Logistic Regression | ~1s | ~15 min |
| 3. Evaluate | PR curves, confusion matrices | ~3s | ~5 min |

### Memory Usage

| Data | Shape | Memory |
|------|-------|--------|
| X.npy (raw) | (934762, 1000, 3) | ~11 GB (mmap) |
| handcraft.npy | (934762, 32) | ~120 MB |
| minirocket.npy | (934762, 9996) | ~37 GB ⚠️ |

> 💡 Use `--skip-minirocket` if you have <32GB RAM

---

## 📁 Output Files

```
artifacts/gait_filter/
├── features/
│   ├── V.npy                    # ENMO time series
│   ├── Y_binary.npy             # Binary labels (walking=1)
│   ├── split_indices.npz        # Train/test split
│   └── handcraft.npy            # Hand-crafted features (32-dim)
├── models/
│   ├── handcraft_rf.pkl         # Random Forest
│   ├── handcraft_xgb.pkl        # XGBoost
│   └── gait_filter_best.pkl     # Best model (auto-selected)
└── plots/
    ├── pr_curves_comparison.png
    ├── calibration_curves.png
    └── confusion_matrix_*.png

experiments/gait_filter/
├── logs/{project_id}_*.log      # Execution logs
└── {project_id}_final_report.md # Summary report
```

---

## 🐛 Troubleshooting

### Error: "No such file or directory: prepared_data/X.npy"
- **Cause**: Running from wrong directory
- **Fix**: Run from project root: `cd capture24-master`

### Error: "MemoryError" during MiniRocket
- **Cause**: MiniRocket needs ~40GB RAM
- **Fix**: Use `--skip-minirocket` or `--max-samples 100000`

### Warning: "pyts not available"
- **Cause**: Optional dependency missing
- **Fix**: `pip install pyts` (optional, pipeline works without it)

### XGBoost "use_label_encoder" warning
- Safe to ignore, doesn't affect results

---

## 📐 ENMO Calculation

**ENMO** (Euclidean Norm Minus One) - standard metric for wearable accelerometry:

$$\text{ENMO} = \max\left(\sqrt{x^2 + y^2 + z^2} - 1g, 0\right)$$

| Activity | ENMO (g) |
|----------|----------|
| Sleep/Stationary | 0 - 0.02 |
| Sedentary | 0.02 - 0.05 |
| Light Activity | 0.05 - 0.10 |
| **Walking** | **0.10 - 0.30** |
| Running | 0.30 - 1.00+ |

---

## 📚 References

- [CAPTURE-24 Dataset](https://github.com/OxWearables/capture24)
- [UK Biobank Accelerometer Analysis](https://github.com/activityMonitoring/biobankAccelerometerAnalysis)
