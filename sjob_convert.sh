#!/bin/bash

#SBATCH --job-name=VLQlassification
#SBATCH --output=log/std_m1_convert.out
#SBATCH --error=log/std_m1_convert.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --mail-type=ALL
#SBATCH --mail-user=j-e.choi@ip2i.in2p3.fr

python3 tree2hdf.py -d
wait
python3 tree2hdf.py -m
wait
python3 tree2hdf.py -r
