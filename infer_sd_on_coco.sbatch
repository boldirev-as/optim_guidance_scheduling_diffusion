#! /bin/bash
#SBATCH --job-name="test_different_diffusion_guidances"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=150:00:00
#SBATCH --mail-user=alseboldyrev@edu.hse.ru
#SBATCH --mail-type=FAIL
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --output="exp/finetune_%j.out"

module purge
module load Python
module load CUDA/12.2
module load gnu10/10.1

conda activate my_diffusers_new

export WANDB_MODE=offline

accelerate config default
accelerate launch --multi_gpu --num_processes 2 /home/alseboldyrev/GuidanceScheduling_diffusers/finetune.py

#python /home/alseboldyrev/GuidanceScheduling_diffusers/finetune.py
