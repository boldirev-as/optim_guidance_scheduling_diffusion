#! /bin/bash
#SBATCH --job-name="draftk2g"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output="exp/draftk_eval2gpu_%j.out"
#SBATCH --export=ALL

module purge
module load Python
module load CUDA/12.2
module load gnu10/10.1

conda activate ml_new

export WANDB_MODE=offline
export WANDB_DIR=wandb_offline/
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

unset http_proxy HTTPS_PROXY HTTP_PROXY https_proxy ALL_PROXY all_proxy

torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --config-name draft_k_fulltraj_longeval_2gpu_clipfocus
