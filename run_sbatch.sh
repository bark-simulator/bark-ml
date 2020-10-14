#!/bin/bash
#SBATCH --qos normal
#SBATCH -c 4
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB

srun singularity exec -B $PWD:/bark-ml bark_ml.img /bin/bash /bark-ml/bark-ml/run_exp.sh --exp_json /bark-ml/bark-ml/experiments/configs/highway_gcn_small.json --mode train 