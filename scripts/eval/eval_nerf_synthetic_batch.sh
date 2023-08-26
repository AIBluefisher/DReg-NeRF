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
num_blocks=(3 4 3 4 3 2 2 2)

cd $CODE_ROOT_DIR

SPLIT='val_all'

for((i=0;i<${#scenes[@]};i++));
do
    if [ "${MULTI_BLOCKS}" == "true" ]
    then
        for((k=0;k<2;k++));
        do
            CKPT_PATH=$DATA_ROOT_DIR/${DATASET}/out/${scenes[i]}/block_${k}/model.pth
            python eval_ngp_nerf.py \
                --dataset ${DATASET} \
                --root_dir ${DATA_ROOT_DIR}/${DATASET} \
                --scene ${scenes[i]} \
                --expname ${scenes[i]} \
                --factor 1 \
                --multi_blocks \
                --ckpt_path $CKPT_PATH
        done
    else
        CKPT_PATH=$DATA_ROOT_DIR/${DATASET}/out/${scenes[i]}/model.pth
        python eval_ngp_nerf.py \
            --dataset ${DATASET} \
            --root_dir ${DATA_ROOT_DIR}/${DATASET} \
            --scene ${scenes[i]} \
            --expname ${scenes[i]} \
            --factor 1 \
            --ckpt_path $CKPT_PATH
    fi
done
