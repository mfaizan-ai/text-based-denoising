#!/bin/bash
#SBATCH --job-name=windowseat_testing
#SBATCH --output=runs/testv1_0/slurm_%j.out
#SBATCH --error=runs/testv1_0/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mfaizan@college.tcd.ie

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate windowseat

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT=/lustre/disk/home/users/mfaizan/windowseat-reflection-removal/text-based-denoising

cd $PROJECT

mkdir -p runs/testv1_0

# ── Print job info (shows up in .out log) ─────────────────────────────────────
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Started at  : $(date)"
echo "---"

# ── Testing ───────────────────────────────────────────────────────────────────
python -u test.py \
    --checkpoint  runs/trainingv1.0/checkpoint_latest.pt \
    --data-root   dataset/ \
    --meta-dir    dataset_metadata/ \
    --embed-dir   text/text_embeddings/ \
    --output-dir  runs/testv1_0 \
    --resolution  512 \
    --batch-size  8 \
    --num-workers 8 \
    --num-vis     4

echo "---"
echo "Finished at : $(date)"