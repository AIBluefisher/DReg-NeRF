#!/bin/sh

if [ $# -eq 0 ]
then
    START_INDEX=0
    END_INDEX=-1
elif [ $# -eq 1 ]
then
    START_INDEX=$1
    END_INDEX=-1
elif [ $# -eq 2 ]
then
    START_INDEX=$1
    END_INDEX=$2
fi

NUM_VIEWS=120

HOME_DIR=$HOME
DATA_ROOT_DIR=$HOME/'Datasets'/'objaverse'

python download_objaverse.py \
    --output_dir $DATA_ROOT_DIR \
    --start_index $START_INDEX \
    --end_index $END_INDEX


python bpy_render_views.py \
    --json_path $DATA_ROOT_DIR/obj_name_path_${START_INDEX}-${END_INDEX}.json \
    --output_path $DATA_ROOT_DIR \
    --num_views $NUM_VIEWS \
    --resolution 800 800 \
    --device cuda
