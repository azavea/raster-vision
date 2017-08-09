#!/bin/bash

# Train SSD Mobilenet using TF Object Detection API on Pets dataset

CONFIG=$1
# CONFIG=configs/ssd_mobilenet_v1_pets.config
RUN=$2
# RUN="pets0"
SYNC_INTERVAL="15m"

cd /opt/src/detection

# sync results of previous run just in case it crashed in the middle of running
rm -R /opt/data/results/detection/$RUN
aws s3 sync s3://raster-vision/results/detection/$RUN /opt/data/results/detection/$RUN

# download data and model and unzip
aws s3 cp s3://raster-vision/datasets/detection/models/ssd_mobilenet_v1_coco_11_06_2017.zip /opt/data/datasets/detection/models/
unzip -o /opt/data/datasets/detection/models/ssd_mobilenet_v1_coco_11_06_2017.zip -d /opt/data/datasets/detection/models/

aws s3 cp s3://raster-vision/datasets/detection/pets.zip /opt/data/datasets/detection/
unzip -o /opt/data/datasets/detection/pets.zip -d /opt/data/datasets/detection/

/opt/src/detection/scripts/s3_sync.sh $SYNC_INTERVAL $RUN &

# run 3 tf scripts including tensorboard on port 6006
mkdir -p /opt/data/results/detection/$RUN
mkdir -p /opt/data/results/detection/$RUN/train
mkdir -p /opt/data/results/detection/$RUN/eval

python models/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=$CONFIG \
    --train_dir=/opt/data/results/detection/$RUN/train \
& \
python models/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=$CONFIG \
    --checkpoint_dir=/opt/data/results/detection/$RUN/train \
    --eval_dir=/opt/data/results/detection/$RUN/eval \
& \
tensorboard --logdir=/opt/data/results/detection/$RUN/
