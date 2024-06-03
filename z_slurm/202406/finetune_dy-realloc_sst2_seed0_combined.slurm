#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su114

source ~/torch/bin/activate

srun python -m main train-dyrealloc --config ./configs/opt_lora_SST2_350m_grad-norm.toml \
--learning-rate 1e-4 --seed 0
