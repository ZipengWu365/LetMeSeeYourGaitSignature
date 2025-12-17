# HMM-based Mamba for 4-State Wearable Activity Sequence Classification  
Model Design and Experiment Plan (for Code Agent Implementation)

## 0. Non-Negotiable Execution Rules (Code Agent Contract)
1. Do not change the problem definition, dataset semantics, label definitions, windowing, grouping, split protocol, or evaluation metrics.  
2. Do not silently “improve” the design. Implement exactly what is specified here. If an implementation detail is ambiguous, surface it explicitly in the final report as an “Assumption” and keep the default conservative.  
3. Do not add new baselines, remove baselines, or alter baselines’ hyperparameters unless the plan explicitly instructs a sweep.  
4. Do not refactor unrelated code. Only touch files required to implement the new model, training, inference, and evaluation.  
5. Every experiment run must be reproducible: fixed seeds, logged configs, saved artifacts, deterministic splits.  
6. Every result must be reported with the exact same metric computation code path across models.  

## 1. Objective and Hypothesis
### Objective
Build a classification pipeline that matches the data’s structure: “noisy per-window observations” plus “strong temporal constraints between 4 discrete states”.

### Hypothesis
A small Mamba encoder used as an emission model (per-window logits) combined with an explicit HMM (or HSMM) decoder will outperform a “Mamba-on-RF-probabilities” smoother, and be competitive with (or surpass) RF+HMM under strict participant-level splits.

## 2. Data Assumptions (Must Match Existing Pipeline)
This design assumes the current dataset is organized as:
1. Input features per time step/window: `X_feats[t] ∈ R^32`  
2. Labels per time step/window: `y[t] ∈ {0,1,2,3}` (4 states)  
3. Group identifier: `group_id[t]` such that each participant forms one or more contiguous sequences  
4. Sequence boundaries are defined by participant and time order; no cross-participant transitions are allowed  

If sequences are stored as flattened arrays, the implementation must reconstruct sequences using `(group_id, time_index)` ordering exactly as current baselines do.

## 3. Baselines to Preserve (No Changes)
### Baseline B1: RF + HMM (Existing Best)
1. RF trained on 32-D features  
2. Use OOB decision function probabilities on training to fit emissions or smoothing (as currently implemented)  
3. Transition matrix computed from true labels within training groups  
4. Inference uses Viterbi decoding per participant sequence  

### Baseline B2: ESN Smoother (Existing)
No architectural changes. Only rerun via the unified evaluation harness.

### Baseline B3: Mamba Smoother on RF Probabilities (Existing)
No architectural changes. Only rerun via the unified evaluation harness.

All comparisons must be apples-to-apples: same splits, same metric code, same label mapping.

## 4. Proposed Models (New Work)

### Model P1: Tiny-Mamba Emission plus HMM Decode (Primary)
This is the direct analog of RF+HMM, replacing RF with a small, learnable sequence encoder that sees the original 32-D features.

#### P1 Architecture
1. Input: sequence `X ∈ R^{T×32}`  
2. Feature projection: `Linear(32 → d_model)`  
3. Tiny Mamba stack: `n_layers ∈ {1,2}`  
4. Output head: `Linear(d_model → 4)` producing per-time-step logits `e_t`  
5. Emission probabilities: `p_t = softmax(e_t / τ)` with optional temperature `τ`  
6. HMM decode: use transition matrix `A` and emissions `p_t` to compute Viterbi path `ŷ_{1:T}`  

#### P1 HMM Details
1. States: 4 (exactly the label set)  
2. Transition matrix: learned from training labels only, using the same method as baseline `compute_transition(Y, labels, groups)`  
3. Emission model: provided by neural network outputs `p_t` (HMM does not refit emissions)  
4. Decoding: Viterbi per participant sequence  
5. Optional: forward-backward for diagnostics only (state posteriors); not required for inference  

#### P1 Training Objective
Training is discriminative at the emission level:
1. Loss: per-step cross entropy `CE(y_t, e_t)` averaged over all time steps in all training sequences  
2. Class weighting: optional, only if label imbalance is present and already handled elsewhere; if used, it must be logged in config  
3. Early stopping: validation decoded Macro-F1 (see Section 6.4)  

#### Why P1 Fits This Data
1. Preserves explicit transition prior advantage of HMM  
2. Avoids information bottleneck: uses full 32-D features, not 4-D RF probabilities  
3. Keeps capacity small to match participant count and sequence length constraints  

### Model P2: Tiny-Mamba Emission plus HSMM Decode (Optional Upgrade)
Implement only after P1 is fully working and reproduced.

HSMM adds explicit duration modeling; this can be beneficial for sleep/sitting durations.

#### P2 HSMM Details
1. States: 4  
2. Duration distribution per state: start with Poisson or a discrete categorical with max duration `D_max` (derived from training percentiles)  
3. Fit duration parameters from training label runs only  
4. Decode with HSMM Viterbi  

Proceed with P2 only if P1 is within approximately 2–3% Macro-F1 of RF+HMM or better, and P1 is numerically stable.

## 5. Expected Results and Contingency Plan (Performance Playbook)

### Expected Outcomes (Pre-Registered)
1. P1 should beat “Mamba on RF probabilities” (removes 4-D bottleneck; reduces over-parameterization).  
2. P1 should approach RF+HMM; it may exceed RF+HMM if RF underfits emissions or neural emissions calibrate better.  
3. P2 may beat HMM variants if durations matter and estimation is stable.  

### If P1 Underperforms RF+HMM by More Than 3% Macro-F1
Run the steps in order; do not change the core model definition.

#### Step A: Verify Protocol Integrity (Most Common Failure Source)
1. Confirm participant-level split integrity (no leakage across train/val/test)  
2. Confirm transition matrix `A` computed only from training labels  
3. Confirm sequence ordering within each participant is correct  
4. Confirm metric code matches baselines exactly (Macro-F1, label mapping, masking)  

Fix integrity issues and rerun before any tuning.

#### Step B: Emission Calibration Checks
1. Compute “raw” Macro-F1 on argmax of per-step logits (no decoding)  
2. Compute “decoded” Macro-F1 after HMM Viterbi  
3. If raw is strong but decoded is worse, emissions are likely miscalibrated or transition matrix is too sharp  

Actions (tune on validation only):
1. Temperature scaling: search `τ ∈ {0.7, 1.0, 1.3, 1.6}`  
2. Emission floor: `p_t ← (1-ε)p_t + ε/4`, with `ε ∈ {0.00, 0.01, 0.02}`  

#### Step C: Reduce Capacity if Overfitting (Typical on ≈100 Participants)
1. `d_model`: 16 first; then 32. Avoid 64 or 128.  
2. `n_layers`: 1 first; consider 2 only if clear underfitting.  
3. `dropout`: `{0.1, 0.2, 0.3}`  
4. `weight_decay`: `{1e-4, 1e-3}`  
5. `epochs`: cap at 20–30 with early stopping  

Rule: If training Macro-F1 rises while validation Macro-F1 drops, reduce capacity and increase regularization; do not increase layers.

#### Step D: Increase Stability if Training Is Noisy
1. Lower learning rate: `1e-3 → 3e-4 → 1e-4`  
2. Gradient clipping: keep, but log the clip value  
3. Scheduler: only use cosine scheduler if constant LR is unstable; otherwise keep constant LR for simplicity  

### If P1 Underperforms RF+HMM by 3% Macro-F1 or Less
Proceed with the sweep plan in Section 6.5 only; do not change structure.

### If P1 Outperforms RF+HMM
Run confirmatory ablations:
1. P1 raw vs P1 decoded (benefit attributable to explicit transitions)  
2. RF only vs RF+HMM vs P1+HMM under the same evaluation harness  
3. Per-class F1 analysis (which states improved and why)  

## 6. Experiment Design

### 6.1 Split Protocol (Hard Requirement)
Participant-level splitting only; no window-level random splits.
1. Train/Val/Test are disjoint in participant IDs  
2. Transition matrix `A` and HSMM duration parameters (if any) computed from Train only  
3. Calibration (`τ`, `ε`) tuned on Val only  
4. Test used once per configuration family for final reporting  

### 6.2 Metrics (Must Match Baselines Exactly)
1. Primary: Macro F1 across all time steps in the test set  
2. Secondary: per-class F1, confusion matrix  
3. Optional diagnostics: run-length distributions, transition counts, but only as additional analysis  

### 6.3 Evaluation Modes (For Every Proposed Model)
Report both:
1. Raw: argmax of per-step probabilities (no temporal decoding)  
2. Decoded: HMM/HSMM decoded path  

### 6.4 Early Stopping Rule
Use validation decoded Macro-F1 for P1/P2, because inference is decoded.

### 6.5 Hyperparameter Sweep Plan (Pre-Registered and Budgeted)
The sweep is intentionally small to prevent overfitting to validation.

Grid for P1:
1. `d_model ∈ {16, 32}`  
2. `n_layers ∈ {1, 2}`  
3. `dropout ∈ {0.1, 0.2, 0.3}`  
4. `lr ∈ {1e-3, 3e-4, 1e-4}`  
5. `weight_decay ∈ {1e-4, 1e-3}`  
6. `τ = 1.0` initially; tune `τ` only if Step B triggers  

Run budget and staging:
1. Stage 1: fix `dropout=0.2`, `weight_decay=1e-4`, `τ=1.0`; sweep `d_model × n_layers × lr` (12 runs)  
2. Stage 2: take top 3 configs from Stage 1; sweep `dropout × weight_decay` (6 runs each, 18 total)  
3. Stage 3: only if needed, tune `τ` and `ε` on the single best config  

### 6.6 Statistical Reporting
1. Report fixed test split performance  
2. Optionally repeat with 3 different random participant splits only if the codebase already supports this without major refactor  

## 7. Software Engineering Plan (Implementation Specification)

### 7.1 Repository Structure (Additive, Minimal Disruption)
Create new modules without breaking existing baselines:
1. `models/tiny_mamba_emission.py` (P1 emission network)  
2. `models/hmm_decode.py` (wrapper around existing HMM viterbi using provided emissions)  
3. `models/hsmm_decode.py` (optional P2)  
4. `train/train_p1_mamba_hmm.py` (trainer + inference pipeline)  
5. `eval/evaluate_sequence_labeling.py` (single source of truth for metrics)  
6. `configs/` (YAML/JSON configs for sweeps)  
7. `scripts/` (CLI entry points)  
8. `artifacts/` (checkpoints, logs, predictions, decoded paths)  

### 7.2 Single Unified Evaluation Harness (Hard Requirement)
Implement one evaluation function used by all models:
1. Input: per-time-step predictions plus metadata for grouping and masking  
2. Output: Macro-F1, per-class F1, confusion matrix  
3. Must be invoked identically for RF+HMM, ESN, Mamba smoother, and P1/P2  

### 7.3 Configuration Management
All runs must be config-driven:
1. Save exact config file into run directory  
2. Save seed values  
3. Save git commit hash if available  
4. Save predictions (raw probabilities if possible) and decoded labels per participant  

### 7.4 Logging and Artifacts (Per Run)
Save:
1. `metrics.json` (Macro-F1, per-class F1, confusion matrix)  
2. `predictions.parquet` or `predictions.npz` with: `group_id, t, y_true, y_raw_pred, y_decoded_pred, proba_raw[4]`  
3. `transition_matrix.npy` used for decoding  
4. Training curves: loss, raw Macro-F1, decoded Macro-F1 on validation  

### 7.5 Unit Tests (Minimum Set)
1. HMM decode sanity: one-hot emissions return identical labels after decoding  
2. Transition matrix computation: rows sum to 1, no NaNs  
3. Split integrity: no participant ID overlap across splits  

### 7.6 Implementation Constraints for P1
1. P1 must accept full 32-D features (not RF probabilities)  
2. P1 must output per-step logits for 4 states  
3. HMM decode must operate per participant sequence; no cross-boundary transitions  
4. Do not implement new feature engineering or label smoothing unless explicitly invoked by the contingency plan steps  

## 8. Deliverables Checklist (Required Outputs)
1. P1 implementation with training and inference runnable end-to-end  
2. Baselines rerun under the unified evaluation harness  
3. A single markdown report `REPORT.md` containing:
   1. Split protocol summary  
   2. Baseline results (RF+HMM, ESN, Mamba smoother)  
   3. P1 raw vs decoded results  
   4. Best config and a table of attempted configs  
   5. Diagnostics and which contingency steps were triggered (if any)  
4. All artifacts saved under a deterministic run directory naming convention  

## 9. Default Configuration for First Working Run (P1 Smoke Test)
1. `d_model = 16`  
2. `n_layers = 1`  
3. `dropout = 0.2`  
4. `lr = 3e-4`  
5. `weight_decay = 1e-4`  
6. `epochs = 20` with early stopping on validation decoded Macro-F1  
7. `τ = 1.0`, `ε = 0.0`  
8. Gradient clip = 1.0 (log it)  

Acceptance criterion:
Training runs without divergence and produces decoded Macro-F1 meaningfully above chance. Only then proceed to sweep.

## 10. Fairness and Interpretation Notes
1. RF+HMM’s OOB probabilities are a strong anti-overfitting design; P1 must be tuned conservatively to avoid validation overfitting.  
2. Any improvement over RF+HMM must be verified as coming from emissions quality, not leakage or modified transitions.  
3. If P1 matches RF+HMM within noise while providing differentiable emissions and a cleaner extension path (HSMM, covariates), that is still a meaningful outcome.
