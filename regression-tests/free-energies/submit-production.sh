#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J absolv[1-81]
#BSUB -W 12:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:
#
#BSUB -M 4

# Enable conda
. ~/.bashrc

conda activate absolv

export ABSOLV_VERSION=$(python -c "import absolv; print(absolv.__version__)")
conda env export > "absolv-$ABSOLV_VERSION-env.yml"

# Launch my program.
module load cuda/10.1
export OPENMM_CPU_THREADS=1

export SYSTEMS=(
"methane"
"methanol"
"ethane"
"toluene"
"neopentane"
"2-methylfuran"
"2-methylindole"
"2-cyclopentanylindole"
"7-cyclopentanylindole"
)
export N_SYSTEMS=${#SYSTEMS[@]}

export METHODS=("eq-indep" "eq-repex" "neq")
export N_METHODS=${#METHODS[@]}

export REPLICAS=(1 2 3)
export N_REPLICAS=${#REPLICAS[@]}

export INDEX=$(($LSB_JOBINDEX-1))

export SYSTEM_INDEX=$(( INDEX % N_SYSTEMS ))
export METHOD_INDEX=$(( (INDEX / N_SYSTEMS) % N_METHODS ))
export REPLICA_INDEX=$(( INDEX / (N_SYSTEMS * N_METHODS) ))

cp -a "absolv-$ABSOLV_VERSION/" "absolv-$ABSOLV_VERSION-${REPLICAS[REPLICA_INDEX]}/"

export RUN_DIR="absolv-$ABSOLV_VERSION-${REPLICAS[REPLICA_INDEX]}/${METHODS[$METHOD_INDEX]}/${SYSTEMS[$SYSTEM_INDEX]}"
echo $RUN_DIR

export ABSOLV_PACKMOL_SEED=${REPLICAS[REPLICA_INDEX]}

python run-production.py $RUN_DIR
