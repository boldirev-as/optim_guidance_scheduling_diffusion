#! /bin/bash
#SBATCH --job-name="refl_smoke"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output="exp/refl_smoke_%j.out"
#SBATCH --export=ALL

module purge
module load Python
module load CUDA/12.2
module load gnu10/10.1

conda activate ml_new

export WANDB_MODE=offline
export WANDB_DIR=wandb_offline/
export HYDRA_FULL_ERROR=1

unset http_proxy HTTPS_PROXY HTTP_PROXY https_proxy ALL_PROXY all_proxy

srun python train.py --config-name selfconsist_smoke
