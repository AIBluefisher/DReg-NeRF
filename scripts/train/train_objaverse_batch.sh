#!/usr/bin/env bash

MULTI_BLOCKS=$1 # ["true", "false"]
CUDA_IDS=$2 # {'0,1,2,...'}

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/DReg-NeRF'
JSON_DIR=$CODE_ROOT_DIR/'conerf/datasets/register'
DATA_ROOT_DIR=$HOME/'SSD/datasets/dreg_nerf'

DATASET='objaverse'

# AABB="-0.5 -0.5 -0.5 0.5 0.5 0.5"

cd $CODE_ROOT_DIR

SPLIT='train'

if [ "${MULTI_BLOCKS}" == "true" ]
then
    python train_ngp_nerf.py \
        --train_split ${SPLIT} \
        --dataset ${DATASET} \
        --root_dir ${DATA_ROOT_DIR}/$DATASET/'images' \
        --data_split_json ${JSON_DIR}/obj_id_names.json \
        --multi_blocks \
        --min_num_blocks 2 \
        --max_num_blocks 2 \
        --max_iterations 10000 \
        --n_validation 10000 \
        --n_checkpoint 10000 \
        --factor 1 \
        --enable_tensorboard
else
    python train_ngp_nerf.py \
        --train_split ${SPLIT} \
        --dataset ${DATASET} \
        --root_dir ${DATA_ROOT_DIR}/$DATASET/'images' \
        --data_split_json ${JSON_DIR}/obj_id_names.json \
        --max_iterations 15000 \
        --factor 1 \
        --enable_tensorboard
fi
