# LetMeSeeYourGaitSignature

Walking State Recognition from Wearable Accelerometer Data  
**HMM vs ESN vs Mamba** - Comparing temporal smoothing approaches with **CUDA GPU Acceleration**

---

##  CUDA GPU Acceleration

| Model | GPU Support | Implementation | Speedup |
|-------|-------------|----------------|---------|
| HMM |  CPU only | hmmlearn | - |
| ESN |  **CUDA** | PyTorch (reservoir + Ridge) | ~3-5x |
| Mamba |  **CUDA** | mamba-ssm native kernels | ~5-10x |

---

##  Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# PyTorch with CUDA 12.1 (recommended for GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Optional: Mamba SSM (requires Linux + CUDA 11.8+)
pip install mamba-ssm causal-conv1d>=1.1.0
```

### 2. Prepare Data

Place the following files in `prepared_data/` directory:
- `X_feats.pkl` - Extracted features (N, 32) - **192MB**
- `Y_Walmsley2020.npy` - Activity labels - **61MB**
- `P.npy` - Participant IDs - **14MB**

### 3. Run Experiments

```bash
# Quick mode (~20 min CPU / ~8 min GPU): HMM + 2 ESN configs
python run_hmm_esn_mamba_walking_recognition_experiments.py --mode quick

# Standard mode (~40 min CPU / ~15 min GPU): HMM + ESN + Mamba-Light
python run_hmm_esn_mamba_walking_recognition_experiments.py --mode standard

# Full mode (~2 hours CPU / ~40 min GPU): All experiments
python run_hmm_esn_mamba_walking_recognition_experiments.py --mode full
```

---

##  File Structure

```
 run_hmm_esn_mamba_walking_recognition_experiments.py  # Main experiment script
 train_baseline.py            # RF + HMM baseline
 train_esn_smoother.py        # ESN (Echo State Network) with CUDA
 train_mamba_smoother.py      # Mamba SSM with CUDA
 evaluate_smoothers.py        # Model evaluation & comparison
 classifier.py                # Classifier utilities
 hmm.py                       # HMM implementation
 fix_pickle_compat.py         # NumPy pickle compatibility fix
 requirements.txt             # Python dependencies
 MAMBA_WALKING_RECOGNITION.md # Detailed documentation
 readmeforgaitfilter.md       # Gait filter pipeline docs (detailed)
 prepared_data/               # Data files (Git LFS)
     X_feats.pkl              # Features
     Y_Walmsley2020.npy       # Labels
     P.npy                    # Participant IDs
```

---

##  Experiment Modes

| Mode | Duration (CPU) | Duration (GPU) | Experiments |
|------|----------------|----------------|-------------|
| quick | ~20 min | ~8 min | HMM + 2 ESN configs |
| standard | ~40 min | ~15 min | HMM + ESN + Mamba-Light |
| full | ~2 hours | ~40 min | All 7 experiment combinations |

---

##  Requirements

- Python 3.9+
- **PyTorch 2.0+** (required for CUDA acceleration)
- NVIDIA GPU recommended (CUDA 11.8+ for Mamba)
- ~16GB RAM minimum

### Check CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

##  Documentation

- **`MAMBA_WALKING_RECOGNITION.md`** - Theoretical background, architecture comparisons
- **`readmeforgaitfilter.md`** - Detailed gait filter pipeline documentation

---

##  CUDA Implementation Details

### ESN (Echo State Network)
- PyTorch-based reservoir state computation
- GPU-accelerated Ridge regression (closed-form solution)
- Automatic CPU/GPU device selection

### Mamba SSM
- Native CUDA kernels from `mamba-ssm` package
- Full training loop with AdamW + CosineAnnealingLR
- Gradient clipping for stable training

---

##  Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | GPU memory insufficient | Reduce batch size or use CPU |
| `mamba_ssm not found` | Package not installed | `pip install mamba-ssm` (Linux only) |
| `numpy._core` error | Pickle compatibility | Run `fix_pickle_compat.py` |
