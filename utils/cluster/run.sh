#!/bin/bash
#SBATCH --qos lowprio
#SBATCH -c 4
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB

srun singularity exec --nv ../images/bark_ml.img python3 -u ./configuration 