#!/bin/bash

#SBATCH --job-name=m1_VLQlassifciation
#SBATCH --output=log/std_aftertraining.out
#SBATCH --error=log/std_aftertraining.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=j-e.choi@ip2i.in2p3.fr

python3 afterTraining.py -v
python3 afterTraining.py -c
python3 afterTraining.py -i
