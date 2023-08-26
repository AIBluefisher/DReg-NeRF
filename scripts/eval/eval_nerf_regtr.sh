#!/usr/bin/env bash

DATASET=$1
SPLIT=$2
CUDA_IDS=$3

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=20
export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/DReg-NeRF'
JSON_DIR=$CODE_ROOT_DIR/'conerf/datasets/register'
DATA_ROOT_DIR=$HOME/'SSD/datasets/dreg_nerf'

cd $CODE_ROOT_DIR

python eval_nerf_regtr.py \
    --root_dir ${DATA_ROOT_DIR} \
    --json_dir ${JSON_DIR} \
    --dataset ${DATASET} \
    --expname 'train_nerf_regtr_objaverse' \
    --train_split $SPLIT
