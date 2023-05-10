#!/bin/bash

#sbatch sjob_convert.sh
#sbatch -t 0-01:00 -n 1 --gres=gpu:v100:1 --mem 10G sjob_training.sh
sbatch sjob_training.sh
