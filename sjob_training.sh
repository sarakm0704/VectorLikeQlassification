#!/bin/bash

#SBATCH --job-name=VLQlassifciation
#SBATCH --output=log/std_m1_training.out
#SBATCH --error=log/std_m1_training.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=j-e.choi@ip2i.in2p3.fr

python3 train_dnn.py
