#!/usr/bin/env bash

FILE="Dataset-Crawler"
TARGET_DIR=./datasets/$FILE/
if [ ! -d "$TARGET_DIR" ]; then
    URL=https://www.dropbox.com/s/ddyf3s3qm8awtfm/Dataset-Crawler.tar.gz?dl=0
    COMPR_FILE=./datasets/$FILE.tar.gz

    mkdir -p ./datasets
    wget -N $URL -O $COMPR_FILE
    tar -xvzf $COMPR_FILE -C ./datasets/
    rm $COMPR_FILE
fi


GPU=0
export CUDA_VISIBLE_DEVICES=$GPU
python train.py \
--name object_detector_net_prob_exp1 \
--model object_detector_net_prob \
--checkpoints_dir ./checkpoints/object_detector_net_prob/ \
--batch_size 120 \
--gpu_ids 0 \
--lambda_prob 100 \
--poses_g_sigma 0.6 \
--lr 0.0001 \
--nepochs_no_decay 800 \
--nepochs_decay 200