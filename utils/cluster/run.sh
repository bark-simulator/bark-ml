#!/bin/bash
#SBATCH --qos normal
#SBATCH -c 4
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB

srun singularity exec --nv ../images/barkml_diadem.img python3 -u ./diadem_dqn 