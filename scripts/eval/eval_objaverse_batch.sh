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

cd $CODE_ROOT_DIR

SPLIT='val_all'


if [ "${MULTI_BLOCKS}" == "true" ]
then
    python eval_ngp_nerf.py \
        --dataset ${DATASET} \
        --root_dir ${DATA_ROOT_DIR}/${DATASET}/'images' \
        --data_split_json ${JSON_DIR}/obj_id_names.json \
        --factor 1 \
        --multi_blocks
else
    CKPT_PATH=$DATA_ROOT_DIR/${DATASET}/out/${scenes[i]}/model.pth
    python eval_ngp_nerf.py \
        --dataset ${DATASET} \
        --root_dir ${DATA_ROOT_DIR}/${DATASET}/'images' \
        --data_split_json ${JSON_DIR}/obj_id_names.json \
        --factor 1 \
        --ckpt_path $CKPT_PATH
fi
