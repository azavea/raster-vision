# Object Detection

## Overview

This guide shows how to prepare a training dataset, train a model, make predictions, and evaluate the predictions. This demonstrates a set of scripts which are built on top of the Tensorflow Object Detection API, and AWS Batch to run jobs. Unless otherwise stated, all commands should be run from inside the Docker CPU container. In addition, the paths are with reference to the file system of the Docker CPU container. Note that there are bugs, so make sure to check the issues with the `object-detection` and `bug` labels. If you are not at Azavea, you will need to upload your own data to your own S3 bucket. These scripts handle URIs that can be either local files or S3 URIs.

## Prepare training data

A project is a set of GeoTIFFs covering an area of interest. Given a set of projects, and a set of corresponding annotation GeoJSON files, you can generate a training dataset. Here is an example of how to do so for
Oakland and Richmond ship data. First, create a projects configuration JSON file with the following content, and upload it to S3.

```
[
    {
        "images": [
            "s3://raster-vision/datasets/detection/ship_test/oakland.tiff"
        ],
        "annotations": "s3://raster-vision/datasets/detection/ship_test/2017-07-23-oakland-final.geojson"
    },
    {
        "images": [
            "s3://raster-vision/datasets/detection/ship_test/richmond.tiff"
        ],
        "annotations": "s3://raster-vision/datasets/detection/ship_test/2017-08-27-richmond-final.geojson"
    }
]
```

Next, run the following command which will place the output at `s3://raster-vision/datasets/detection/ship_test_out.zip`. On a larger set of projects, you will probably want to run this on EC2 to avoid downloading a lot of data.

```
python -m rv.run prep_train_data \
    --debug \
    --chip-size 300 --num-neg-chips 50 --max-attempts 500 \
    s3://raster-vision/datasets/detection/ship_test/projects.json \
    s3://raster-vision/datasets/detection/ship_test_out.zip
```

## Train a model

The training algorithm is configured in a file, an example of which can be seen at `samples/configs/ships/ship_test.config`. This file needs to be modified if the local paths for data files change, or to tweak the hyperparameters or model architecture.
After uploading this file to S3, and committing any desired changes to a git branch, start a training job on AWS Batch, by running the following *from the VM*. Note the quotes around the command to run on the container.
```
src/detection/scripts/batch_submit.py <branch-name> \
    "python -m rv.run train \
        --sync-interval 600 \
        s3://raster-vision/results/detection/train/ship_test/ship_test.config \
        s3://raster-vision/datasets/detection/ship_test.zip \
        s3://raster-vision/datasets/detection/models/ssd_mobilenet_v1_coco_11_06_2017.zip \
        s3://raster-vision/results/detection/train/ship_test"
    --attempts 1
```

This requires that there is a model checkpoint file (for using a pre-trained model) at `s3://raster-vision/datasets/detection/ssd_mobilenet_v1_coco_11_06_2017.zip`. This file can be downloaded from the website for the TF Object Detection API.
As training progresses, the checkpoints are periodically sync'd to S3 according to the `sync-interval` (which is in seconds).
You can monitor training using Tensorboard by pointing your browser at `http://<ec2 instance ip>:6006`. It may take a few minutes before results show up. When you are done training the model (ie. after the total loss flattens out), you need to kill the Batch job since it's running in an infinite loop. If this script fails due to instance shutdown and is run again, it should pick up where it left off using saved checkpoints.

## Make predictions

After training finishes, you can make predictions for a project.
First, in order to make the trained model available for inference, you must first convert a checkpoint file into an inference graph. For example, you can do the following.

```
aws s3 cp s3://raster-vision/results/detection/train/ship_test/train/model.ckpt-57662.index /tmp/
aws s3 cp s3://raster-vision/results/detection/train/ship_test/train/model.ckpt-57662.meta /tmp/
aws s3 cp s3://raster-vision/results/detection/train/ship_test/train/model.ckpt-57662.data-00000-of-00001 /tmp/

python models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /opt/src/detection/configs/ships/ship_test.config \
    --checkpoint_path /tmp/model.ckpt-57662 \
    --inference_graph_path

aws s3 cp
/tmp/inference_graph.pb s3://raster-vision/results/detection/train/ship_test/train/
```

Next, upload some TIFF files to S3. To run a prediction job locally on the Oakland project, run the following. (Remember this is "cheating" though since the model was trained on Oakland.)

```
python -m rv.run predict \
    s3://raster-vision/results/detection/train/ship_test/train/inference_graph.pb \
    s3://raster-vision/datasets/detection/ship_test_label_map.pbtxt \
    s3://raster-vision/results/detection/predict/ship_test/oakland.tif \
    s3://raster-vision/results/detection/predict/ship_test/agg_predictions.json
```

When this is finished running, there should be a GeoJSON file with predictions at     `s3://raster-vision/results/detection/predict/ship_test/agg_predictions.json`.

## Evaluate predictions

Aside from qualitatively evaluating the predictions in QGIS, you can quantify how good the predictions are compared to the ground truth using a script as follows. This outputs a JSON file with the precision and recall for each class.

```
python -m rv.run eval_predictions \
    s3://raster-vision/results/detection/predict/ship_test/oakland.tif \
    s3://raster-vision/datasets/detection/ship_test_label_map.pbtxt \
    s3://raster-vision/datasets/detection/jm_ships/2017-07-23-oakland-final.geojson \
    s3://raster-vision/results/detection/predict/ship_test/agg_predictions.json \
    s3://raster-vision/results/detection/predict/ship_test/eval.json
```

## Data representation

Unless otherwise stated, we represent bounding boxes as numpy arrays of shape
`[N, 4]` with `[ymin, xmin, ymax, xmax]` as the columns, following a convention set in the TF Object Detection API. In addition, `y` refers to the index of the columns of an element in an array, and `x` refers to the index of the column.

## Debugging

To debug, it may be helpful to run the above two scripts locally, and inspect the temporary files generated inside `/opt/data/temp`.
