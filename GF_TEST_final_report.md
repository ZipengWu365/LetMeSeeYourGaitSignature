# GF_TEST Gait Filter - Final Report

## Model Comparison

| Model | PR-AUC | F1 | Prec@Recall>=0.70 |
|-------|--------|-----|-------------------|
| handcraft_lr | 0.3778 | 0.3860 | 0.2598 |
| handcraft_rf | 0.3509 | 0.4164 | 0.2546 |
| handcraft_xgb | 0.3112 | 0.3767 | 0.2411 |
| minirocket_lr | 0.3011 | 0.4109 | 0.2836 |
| minirocket_ridge | 0.2323 | 0.3568 | 0.1891 |

## Best Model
- **Model**: handcraft_lr
- **PR-AUC**: 0.3778
- **Model Size**: 0.00 MB
- **Inference Speed**: 0.0003 s per 1000 windows

## Hard Negatives (Walking vs Chores)
- **PR-AUC**: 0.22940555629920176
- **Prec@Recall>=0.70**: 0.1574074074074074

## Artifacts
- PR Curves: `artifacts\gait_filter\plots\pr_curves_comparison.png`
- Calibration: `artifacts\gait_filter\plots\calibration_curves.png`
- Best Model: `artifacts\gait_filter\models\gait_filter_best.pkl`