#! /bin/bash
#SBATCH --job-name="rcsgp"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output="exp/rcsgp_%j.out"
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

export TMPDIR="${SLURM_TMPDIR:-$PWD/tmp}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

unset http_proxy HTTPS_PROXY HTTP_PROXY https_proxy ALL_PROXY all_proxy

srun python train.py --config-name rcsgp dataloader.train.num_workers=0 dataloader.test.num_workers=0
