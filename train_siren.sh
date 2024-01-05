#!/bin/bash
#$ -N sdf
#$ -S /bin/bash
#$ -q gpu_long_2080ti
#$ -cwd
#$ -P kenprj
#$ -t 1-50:1
#$ -pe omp 1
#$ -o ./std_logs2/$JOB_NAME_$TASK_ID.out
#$ -j y

conda deactivate
conda activate piper

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

OBJ_NUM=$(($SGE_TASK_ID-1))

XYZ_DIR=xyz_dir
WEIGHTS_DIR=siren_weights

mkdir -p $XYZ_DIR
mkdir -p $WEIGHTS_DIR

export WANDB_NAME="Mesh ${OBJ_NUM}"

python3 train_siren.py --epochs 5000 --obj_path ./test_task_meshes/${OBJ_NUM}.obj --xyz_path ./${XYZ_DIR}/${OBJ_NUM}.xyz --weights_path ./${WEIGHTS_DIR}/${OBJ_NUM}.pt
