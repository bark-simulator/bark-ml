#!/bin/bash
#SBATCH --qos lowprio
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB

if [ "$EUID" -ne 0 ]
  then echo "Please run as root."
  exit
fi

srun singularity exec --nv -B .:/experiment /mnt/glusterdata/home/hart/images/bark_ml.img python3 -u ./$1.runfiles/configuration 
