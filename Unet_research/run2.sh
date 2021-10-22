#!/bin/bash

#$ -pe smp 8
#$ -q gpu
#$ -l gpu_card=1
#$ -N training
#$ -M jlee88@nd.edu
#$ -m ae

export OMP_NUM_THREADS=${NSLOTS}
module load python

python training.py