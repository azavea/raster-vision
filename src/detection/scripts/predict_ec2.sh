#!/bin/bash

# This takes a GeoTIFF file and makes predictions in GeoJSON format.
# It's meant to be run on EC2 and downloads/uploads relevant files accordingly.

# Parse args
LOCAL=false
DEBUG_FLAG=""
while :; do
    case $1 in
        --config-path)
            CONFIG_PATH=$2
            shift
            ;;
        --train-id)
            TRAIN_ID=$2
            shift
            ;;
        --checkpoint-id)
            CHECKPOINT_ID=$2
            shift
            ;;
        --predict-id)
            PREDICT_ID=$2
            shift
            ;;
        --dataset-id)
            DATASET_ID=$2
            shift
            ;;
        --local)
            LOCAL=true
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)
            break
    esac
    shift
done

echo "CONFIG_PATH     = ${CONFIG_PATH}"
echo "TRAIN_ID        = ${TRAIN_ID}"
echo "CHECKPOINT_ID   = ${CHECKPOINT_ID}"
echo "PREDICT_ID      = ${PREDICT_ID}"
echo "DATASET_ID      = ${DATASET_ID}"
echo "LOCAL           = ${LOCAL}"
echo "DEBUG_FLAG      = ${DEBUG_FLAG}"

set -e -x
cd /opt/src/detection

S3_DATASETS=s3://raster-vision/datasets/detection
LOCAL_DATASETS=/opt/data/datasets/detection

S3_TRAIN=s3://raster-vision/results/detection/train/${TRAIN_ID}
LOCAL_TRAIN=/opt/data/results/detection/train/${TRAIN_ID}

S3_PREDICT=s3://raster-vision/results/detection/predict/${PREDICT_ID}
LOCAL_PREDICT=/opt/data/results/detection/predict/${PREDICT_ID}

TIFF_PATH=${LOCAL_PREDICT}/index.vrt
MASK_PATH=${LOCAL_PREDICT}/mask.json
INFERENCE_GRAPH_PATH=${LOCAL_TRAIN}/inference_graph.pb
LABEL_MAP_PATH=${LOCAL_DATASETS}/${DATASET_ID}/label_map.pbtxt

# Put temp files here for easy inspection.
TEMP_PATH=/opt/data/temp
mkdir -p ${TEMP_PATH}

if [ "$LOCAL" = false ] ; then
    # download tiff to run prediction on
    aws s3 sync ${S3_PREDICT} ${LOCAL_PREDICT}

    # download results of training
    aws s3 sync ${S3_TRAIN} ${LOCAL_TRAIN}

    # download training data and unzip (we just need the label map though...)
    aws s3 cp ${S3_DATASETS}/${DATASET_ID}.zip ${LOCAL_DATASETS}/${DATASET_ID}.zip
    unzip -o ${LOCAL_DATASETS}/${DATASET_ID}.zip -d ${LOCAL_DATASETS}
fi

# convert checkpoint to frozen inference graph
rm -R ${TEMP_PATH}
python models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${CONFIG_PATH} \
    --checkpoint_path ${LOCAL_TRAIN}/train/model.ckpt-${CHECKPOINT_ID} \
    --inference_graph_path ${INFERENCE_GRAPH_PATH}

# run sliding window over tiff to generate lots of window files
python scripts/make_windows.py \
    --image-path ${TIFF_PATH} \
    --output-dir ${TEMP_PATH}/windows \
    --window-size 300

# run prediction on the windows
python scripts/predict.py \
    --frozen-graph-path ${INFERENCE_GRAPH_PATH} \
    --label-map-path ${LABEL_MAP_PATH} \
    --input-dir ${TEMP_PATH}/windows/images \
    --output-dir ${TEMP_PATH}/windows/predictions

# aggregate the predictions into an output geojson file
python scripts/aggregate_predictions.py \
    --image-path ${TIFF_PATH} \
    --window-info-path ${TEMP_PATH}/windows/window_info.json \
    --predictions-path ${TEMP_PATH}/windows/predictions/predictions.json \
    --label-map-path ${LABEL_MAP_PATH} \
    --output-dir ${LOCAL_PREDICT}/output ${DEBUG_FLAG}

if [ -e ${MASK_PATH} ]
then
    mv ${LOCAL_PREDICT}/output/predictions.geojson \
        ${LOCAL_PREDICT}/output/unfiltered_predictions.geojson
    python scripts/filter_geojson.py \
        --mask-path ${MASK_PATH} \
        --input-path ${LOCAL_PREDICT}/output/unfiltered_predictions.geojson \
        --output-path ${LOCAL_PREDICT}/output/predictions.geojson
fi

if [ "$LOCAL" = false ] ; then
    # upload the results to s3
    aws s3 sync ${LOCAL_TRAIN} ${S3_TRAIN}
    aws s3 sync ${LOCAL_PREDICT} ${S3_PREDICT}
fi
