#!/bin/bash
#SBATCH --job-name=windowseat_training
#SBATCH --output=runs/trainingv1.0/slurm_%j.out
#SBATCH --error=runs/trainingv1.0/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mfaizan@college.tcd.ie

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate windowseat

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT=/lustre/disk/home/users/mfaizan/windowseat-reflection-removal/text-based-denoising

cd $PROJECT

mkdir -p runs/trainingv1.0

# ── Print job info (shows up in .out log) ─────────────────────────────────────
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Started at  : $(date)"
echo "---"

# ── Training ──────────────────────────────────────────────────────────────────
python -u train.py \
    --data-root    dataset/ \
    --meta-dir     dataset_metadata/ \
    --embed-dir    text/text_embeddings/ \
    --output-dir   runs/trainingv1.0 \
    --epochs       20 \
    --batch-size   8 \
    --grad-accum   2 \
    --lr           1e-4 \
    --resolution   512 \
    --lambda-lpips 0.1 \
    --num-workers  16 \
    --wandb-project windowseat \
    --run-name     trainingv1.0

echo "---"
echo "Finished at : $(date)"