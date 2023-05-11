#!/bin/bash

#SBATCH --job-name=m1_VLQlassifciation
#SBATCH --output=log/std_training.out
#SBATCH --error=log/std_training.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=j-e.choi@ip2i.in2p3.fr

python3 train_dnn.py
