#!/usr/bin/env bash

MULTI_BLOCKS=$1 # ["true", "false"]
CUDA_IDS=$2 # {'0,1,2,...'}

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/DReg-NeRF'
DATA_ROOT_DIR=$HOME/'SSD/datasets/dreg_nerf'

DATASET='nerf_synthetic'
scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# AABB="-0.5 -0.5 -0.5 0.5 0.5 0.5"

cd $CODE_ROOT_DIR

SPLIT='train'

for((i=0;i<${#scenes[@]};i++));
do
    if [ "${MULTI_BLOCKS}" == "true" ]
    then
        python train_ngp_nerf.py \
            --train_split ${SPLIT} \
            --dataset ${DATASET} \
            --root_dir ${DATA_ROOT_DIR}/$DATASET \
            --scene ${scenes[i]} \
            --expname ${scenes[i]} \
            --multi_blocks \
            --min_num_blocks 2 \
            --max_num_blocks 2 \
            --max_iterations 20000 \
            --factor 1 \
            --enable_tensorboard
    else
        python train_ngp_nerf.py \
            --train_split ${SPLIT} \
            --dataset ${DATASET} \
            --root_dir ${DATA_ROOT_DIR}/$DATASET \
            --scene ${scenes[i]} \
            --expname ${scenes[i]} \
            --max_iterations 20000 \
            --factor 1 \
            --enable_tensorboard
    fi
done
