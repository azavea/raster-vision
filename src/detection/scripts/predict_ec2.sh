#!/bin/bash

CONFIG=$1
# CONFIG=/opt/src/detection/configs/ssd_mobilenet_v1_pets.config
RUN=$2
# RUN="pets0"
CHECKPOINT_NUMBER=$3
# CHECKPOINT_NUMBER=135656

cd /opt/src/detection

# Download pets dataset
aws s3 cp s3://raster-vision/datasets/detection/pets.zip /opt/data/datasets/detection/
unzip -o /opt/data/datasets/detection/pets.zip -d /opt/data/datasets/detection/

aws s3 sync s3://raster-vision/results/detection/$RUN /opt/data/results/detection/$RUN

python models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $1 \
    --checkpoint_path /opt/data/results/detection/$RUN/train/model.ckpt-${CHECKPOINT_NUMBER} \
    --inference_graph_path /opt/data/results/detection/$RUN/inference_graph.pb

python scripts/predict.py \
    --frozen_graph_path=/opt/data/results/detection/$RUN/inference_graph.pb \
    --label_map_path=/opt/data/datasets/detection/pets/pet_label_map.pbtxt \
    --input_dir=/opt/data/datasets/detection/pets/images_subset \
    --output_dir=/opt/data/results/detection/$RUN/predictions

aws s3 sync /opt/data/results/detection/$RUN s3://raster-vision/results/detection/$RUN
