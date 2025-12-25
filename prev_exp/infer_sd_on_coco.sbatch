#! /bin/bash
#SBATCH --job-name="test_different_diffusion_guidances"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --output="exp/finetune_%j.out"

module purge
module load Python
module load CUDA/12.2
module load gnu10/10.1

conda activate ml_new

export WANDB_MODE=offline

srun python main.py
#python /home/alseboldyrev/GuidanceScheduling_diffusers/finetune.py
