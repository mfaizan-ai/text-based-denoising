#!/bin/bash
#SBATCH --job-name=combined
#SBATCH --output=runs/combined_training/slurm_%j.out
#SBATCH --error=runs/combined_training/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mfaizan@college.tcd.ie

# ── Absolute project path — everything is relative to this ────────────────────
PROJECT=/lustre/disk/home/users/mfaizan/windowseat-reflection-removal/text-based-denoising


# ── Run setup script but then FORCE working directory back to project ──────────
# setup_everything.sh changes cwd to home — we override it immediately after
source $PROJECT/../../bash_scripts/setup_everything.sh
cd $PROJECT   # re-set cwd AFTER setup script, not before

# ── Environment ────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate windowseat

# Create output dir before SLURM tries to write logs to it
mkdir -p $PROJECT/runs/combined_training

# ── Job info ───────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Memory      : $SLURM_MEM_PER_NODE MB"
echo "Working dir : $(pwd)"
echo "Started at  : $(date)"
echo "======================================================================"

# ── Sanity checks — catch missing files before wasting GPU time ───────────────
echo "--- Sanity checks ---"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
python -c "from diffusers import QwenImageEditPipeline; print('diffusers OK')"
ls $PROJECT/dataset_metadata/train_metadata.json && echo "train metadata OK"
ls $PROJECT/dataset_metadata/val_metadata.json   && echo "val metadata OK"
ls $PROJECT/dataset_metadata/test_metadata.json  && echo "test metadata OK"
ls $PROJECT/text/text_embeddings/blur/0.pt       && echo "embeddings OK"
ls $PROJECT/train_windowseat.py                             && echo "train.py OK"
echo "---------------------"

# ── Training — all paths absolute ─────────────────────────────────────────────
python -u $PROJECT/train_windowseat.py \
    --data-root    $PROJECT/dataset \
    --meta-dir     $PROJECT/dataset_metadata \
    --embed-dir    $PROJECT/text/text_embeddings \
    --output-dir   $PROJECT/runs/combined_training \
    --use-task-aware-loss \
     --use-multitask-lora \
    --total-steps  11000 \
    --batch-size   2 \
    --grad-accum   1 \
    --lr-start     1e-5 \
    --lr-peak      1e-4 \
    --lr-decay     5e-6 \
    --warmup-steps 100 \
    --resolution   608 \
    --lambda-psnr  0.1 \
    --lambda-ssim  20.0 \
    --lora-rank    128 \
    --lora-alpha   128 \
    --lora-dropout 0.0 \
    --log-interval  50 \
    --val-interval  500 \
    --save-interval 1000 \
    --num-workers  8 \
    --wandb-project combined_training \
    --run-name     combined_training_11k

EXIT_CODE=$?

echo "======================================================================"
echo "Finished at : $(date)"
echo "Exit code   : $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE