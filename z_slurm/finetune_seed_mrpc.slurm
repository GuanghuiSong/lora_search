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

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--seed 13

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--seed 42

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--seed 87

srun python -m main train --config ./configs/opt_lora_mrpc_350m.toml \
--seed 100

srun python -m main train --config ./configs/opt_plain_mrpc_350m.toml \
--seed 13

srun python -m main train --config ./configs/opt_plain_mrpc_350m.toml \
--seed 42

srun python -m main train --config ./configs/opt_plain_mrpc_350m.toml \
--seed 87

srun python -m main train --config ./configs/opt_plain_mrpc_350m.toml \
--seed 100
