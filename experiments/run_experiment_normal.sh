#!/bin/bash
#SBATCH --qos normal
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB

srun singularity exec --nv bark_ml.img python3 -u ./experiments/run_experiment "$@"
