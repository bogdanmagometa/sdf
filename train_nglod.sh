#!/bin/bash
#$ -N sdf
#$ -S /bin/bash
#$ -q gpu_long_2080ti
#$ -cwd
#$ -P kenprj
#$ -pe omp 1
#$ -o ./std_logs_nglod_nn/$JOB_NAME_$TASK_ID.out
#$ -j y
#$ -tc 8
#$ -t 2-50:1

conda deactivate
conda activate nglod

# OBJ_NUM=0
OBJ_NUM=$(($SGE_TASK_ID-1))

# XYZ_DIR=xyz_dir
# WEIGHTS_DIR=siren_weights

# mkdir -p $XYZ_DIR
# mkdir -p $WEIGHTS_DIR

export WANDB_MODE=offline
export WANDB_NAME="Mesh ${OBJ_NUM}"
export WANDB_ENTITY=bohdanmahometa
export WANDB_PROJECT=nglod-4-sdf

python3 train_nglod.py $OBJ_NUM
# python3 train_siren.py --epochs 5000 --obj_path ./test_task_meshes/${OBJ_NUM}.obj --xyz_path ./${XYZ_DIR}/${OBJ_NUM}.xyz --weights_path ./${WEIGHTS_DIR}/${OBJ_NUM}.pt
