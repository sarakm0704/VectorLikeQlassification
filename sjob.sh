#!/bin/bash

#SBATCH --job-name=m1_VLQlassification
#SBATCH --output=log/std.out
#SBATCH --error=log/std.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --mail-type=ALL
#SBATCH --mail-user=j-e.choi@ip2i.in2p3.fr

python3 tree2hdf.py -d
wait
python3 tree2hdf.py -m
wait
python3 tree2hdf.py -r
wait
python3 train_dnn.py
wait
python3 afterTraining.py -v
python3 afterTraining.py -c
python3 afterTraining.py -i

echo "Done!"
