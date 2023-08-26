#!/usr/bin/env bash

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=20

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/DReg-NeRF'
JSON_DIR=$CODE_ROOT_DIR/'conerf/datasets/register'
DATA_ROOT_DIR=$HOME/'SSD/datasets/dreg_nerf'

cd $CODE_ROOT_DIR

if [ $# == 1 ]; then
    CUDA_IDS=$1 # {'0,1,2,...'}

    export CUDA_VISIBLE_DEVICES=${CUDA_IDS}
    
    python train_nerf_regtr.py \
        --root_dir ${DATA_ROOT_DIR} \
        --json_dir ${JSON_DIR} \
        --expname 'train_nerf_regtr_objaverse' \
        --lr 0.0001 \
        --epochs 80 \
        --n_tensorboard 200 \
        --n_validation 1000 \
        --robust_loss \
        --enable_tensorboard
fi

if [ $# == 3 ]; then
    dataset=$1
    scene=$2
    CUDA_IDS=$3 # {'0,1,2,...'}

    export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

    python train_nerf_regtr.py \
        --dataset ${dataset} \
        --root_dir ${DATA_ROOT_DIR}/$dataset \
        --json_dir ${JSON_DIR}/${scene}'.json' \
        --scene ${scene} \
        --expname 'train_nerf_regtr_'${scene} \
        --lr 0.0001 \
        --epochs 80 \
        --n_tensorboard 200 \
        --n_validation 1000 \
        --robust_loss \
        --enable_tensorboard
fi
