# GaitFilter: Walking Detection from Accelerometer Data

A machine learning pipeline for detecting walking/gait from wrist-worn accelerometer data, trained on the CAPTURE-24 dataset.

## 🎯 Overview

This project builds a binary classifier to filter "walking" segments from free-living accelerometer data. Key features:

- **Multiple feature extractors**: Hand-crafted features, MiniRocket, SAX, SFA
- **Multiple classifiers**: Random Forest, XGBoost, Logistic Regression, Ridge, MrSQM
- **Production-ready**: Includes model export, inference speed benchmarks, calibration curves

## 📊 Results (CAPTURE-24)

| Model | PR-AUC | F1 | Inference Speed |
|-------|--------|-----|-----------------|
| **handcraft_lr** | **0.378** | 0.386 | 0.0003s/1000 windows |
| handcraft_rf | 0.351 | 0.416 | 0.066s/1000 windows |
| handcraft_xgb | 0.311 | 0.377 | - |
| minirocket_lr | 0.301 | 0.411 | - |

## 🚀 Quick Start

### Prerequisites

1. **CAPTURE-24 Data**: Download from [Oxford Research Archive](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16f12bcf25)
2. **Preprocess**: Use `capture24-master/prepare_data.py` to generate:
   - `X.npy`: (N, 1000, 3) accelerometer windows
   - `Y.npy`: (N,) activity labels  
   - `P.npy`: (N,) participant IDs

### Run Pipeline (Linux)

```bash
# One-click run
chmod +x run.sh
./run.sh --data-dir /path/to/prepared_data

# Skip MiniRocket for faster testing
./run.sh --data-dir /path/to/prepared_data --skip-minirocket

# Quick test mode (10k samples)
./run.sh --data-dir /path/to/prepared_data --quick-test
```

### Run Pipeline (Windows/Manual)

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py --project-id MY_EXP --data-dir /path/to/prepared_data

# Run specific phase
python run_pipeline.py --phases preprocess extract --data-dir /path/to/prepared_data
```

## 📁 Project Structure

```
GaitFilter/
├── run_pipeline.py      # Main entry point
├── run.sh               # Linux one-click runner
├── preprocess.py        # Phase 0: ENMO, binary labels, group split
├── extract_features.py  # Phase 1: Handcraft, MiniRocket, SAX, SFA
├── train_classifiers.py # Phase 2: Train all classifiers
├── evaluate.py          # Phase 3: PR curves, confusion matrices
├── compute_biomarkers.py# Bonus: Extract gait biomarkers from filtered data
├── requirements.txt     # Dependencies
├── artifacts/           # Output directory
│   ├── features/        # Extracted features (*.npy)
│   ├── models/          # Trained models (*.pkl)
│   └── plots/           # Visualizations (*.png)
└── logs/                # Execution logs
```

## 🔬 Features Extracted

### Hand-crafted (32 features)
- Statistical: mean, std, min, max, range, IQR, skew, kurtosis (per axis)
- ENMO-based: mean ENMO, std ENMO
- Frequency: dominant frequency, spectral entropy (per axis)
- Correlation: inter-axis correlations

### MiniRocket (~10,000 features)
- Random convolutional kernels with PPV aggregation
- Fast and effective for time series classification

## 📝 Citation

If you use this code, please cite the CAPTURE-24 dataset:

```bibtex
@article{willetts2018statistical,
  title={Statistical machine learning of sleep and physical activity phenotypes...},
  author={Willetts, Matthew and Hollowell, Sven and Maywald, Louis and ...},
  journal={Scientific reports},
  year={2018}
}
```

## 📄 License

MIT License
