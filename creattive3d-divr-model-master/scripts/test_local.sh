#!/bin/bash

TEST_NAME=$1

# Paths (CHANGE these paths according to your environment)
DATA_PATH="./demo_data"
DiVR_PATH="../DiVR"
DiVR_RESULTS="./pretrained"

# Test configuration
SLIDE_WIN_EVAL=30
SLIDE_WIN=30
BATCH_SIZE=16
MODEL_TYPE='DiVR_het'
MOTION_DIM=38
CSV_TEST="${TEST_NAME}.csv"

LOAD_MODEL_DIR="${DiVR_RESULTS}/chkpoints_${MODEL_TYPE}/divr_model.pth"
OUTPUT_PATH="${DiVR_PATH}/results_${MODEL_TYPE}/${TEST_NAME}"

 python3 evaluation.py  --save_fre 1 \
 --val_fre 1 --seq_decay_ratio 1 --gaze_points 10 --batch_size 8 --sample_points 300000 \
 --motion_n_layers 6 --gaze_n_layers 6 --gaze_latent_dim 256 --cross_hidden_dim 256 \
 --cross_n_layers 6 --num_workers 8 --output_path $OUTPUT_PATH --load_model_dir $LOAD_MODEL_DIR \
 --vposer_path vposer_v1_0 --smplx_path smplx_models --dataroot $DATA_PATH --data_freq 60 \
 --train_set $CSV_TEST --slide_win $SLIDE_WIN --slide_win_eval $SLIDE_WIN_EVAL --motion_dim $MOTION_DIM \
 --model_type $MODEL_TYPE

