#!/bin/bash

#$ -pe smp 16
#$ -q gpu
#$ -l gpu_card=1
#$ -M jlee88@nd.edu
#$ -m ae
#$ -N optimizer

export OMP_NUM_THREADS=${NSLOTS}
module load python

python optimizer.py 50
python training.py