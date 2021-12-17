#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J absolv
#BSUB -W 00:10
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any cpu options.
#BSUB -q cpuqueue
#
#BSUB -M 4

for REPLICA_INDEX in 1 2 3
do

  export ABSOLV_PACKMOL_SEED=$(( REPLICA_INDEX * 100 ))
  python run-setup.py $REPLICA_INDEX

done