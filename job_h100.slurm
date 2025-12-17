#!/bin/bash
#SBATCH --job-name=gait_full
#SBATCH --partition=swarm_h100              # <<< 改成你们GPU分区名（常见：gpu / nvidia / a100 等）
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:00:00

# GPU 申请（二选一：更推荐先用 --gres 这行）
#SBATCH --gres=gpu:1                 # <<< 申请1张GPU（常见写法）
##SBATCH --gpus=1                    # <<< 如果你们系统要求用这个，把上面 --gres 注释掉并启用此行

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jc15u24@soton.ac.uk

set -euo pipefail

mkdir -p logs
module load cuda
module load gcc
# === 项目根目录 ===
PROJECT_DIR=/iridisfs/scratch/jc15u24/Code/activity/LetMeSeeYourGaitSignature

# === 固定使用你的 conda 环境 python（避免跑到系统 Python）===
PYTHON=/iridisfs/home/jc15u24/.conda/envs/easygen-clean/bin/python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "==================== ENV CHECK ===================="
echo "Host: $(hostname)"
echo "Workdir before cd: $(pwd)"
echo "PYTHON: ${PYTHON}"
ls -l "${PYTHON}"

echo "--- SLURM resources ---"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"

echo "--- Python executable / version ---"
${PYTHON} -c "import sys; print('sys.executable =', sys.executable); print('python version =', sys.version)"

echo "--- GPU / nvidia-smi (if available) ---"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found (maybe AMD GPU or module not loaded)."

echo "--- Torch check ---"
${PYTHON} -c "import torch; print('torch =', torch.__version__); print('torch cuda =', torch.version.cuda); print('cuda available =', torch.cuda.is_available()); print('device count =', torch.cuda.device_count());"

echo "==================================================="

cd "${PROJECT_DIR}"
echo "Workdir after cd: $(pwd)"
echo "Listing top-level files:"
ls -la | head -n 80

echo "==================== RUN =========================="
# === 运行 full 模式（必须用 ${PYTHON}）===
${PYTHON} run_hmm_esn_mamba_walking_recognition_experiments.py --mode full
echo "==================== DONE ========================="
