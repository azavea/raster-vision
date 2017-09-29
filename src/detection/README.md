# Object Detection

## Overview

This guide shows how to train a model on a tiny dataset containing ships, and then how to make predictions on a TIFF file. This makes use of a set of scripts utilizing the Tensorflow Object Detection API, and AWS Batch to run jobs. Unless otherwise stated, all commands should be run from inside the Docker CPU container relative to `/opt/src/detection`. In addition, the paths are with reference to the file system of the Docker CPU container. Note that there are bugs, so make sure to check the issues with the `object-detection` and `bug` labels.

## Data Prep

First, download the raw ships data from  `s3://raster-vision/datasets/detection/singapore_ships.zip` to `/opt/data/datasets/detection/singapore_ships.zip` and unzip it. Then, split
the files into two directories, `train` and `test`. I used all images except `1.tif` and `3.tif` for the training set.

In order to train a model, you must convert a list of GeoTIFFs (corresponding to a VRT) and a GeoJSON file with labels into a set of training chips and a CSV file representing bounding boxes in the chips' frame of reference. You can do this for `0.tif` as follows.

```
python -m rv.run make_train_chips \
    --chip-size 300 --num-neg-chips 50 --max-attempts 500 \
    /opt/data/datasets/detection/singapore_ships/train/0.tif \
    /opt/data/datasets/detection/singapore_ships/train/0.geojson \
    /opt/data/datasets/detection/singapore_ships_chips_neg/chips0 \
    /opt/data/datasets/detection/singapore_ships_chips_neg/chips0.csv
```

You will need to run this command for each VRT in the training set. Next, create a `ships_label_map.pbtxt` file in the `singapore_ships_chips_neg` directory. It should have a single entry with `id=1` and `name=Ships`. An example of this kind of file can be found in `pet_label_map.pbtxt`. Then, convert all sets of training chips to TFRecord format, which is the required format for the TF Object Detection API. The files will be placed in the `singapore_ships_chips_neg` directory, which will include a directory of debug plots if the `--debug` flag is used. To do this for training chip sets 0 and 2, run the following.

```
python -m rv.run make_tf_record --debug \
    /opt/data/datasets/detection/singapore_ships_chips_neg/ships_label_map.pbtxt \
    /opt/data/datasets/detection/singapore_ships_chips_
    /opt/data/datasets/detection/singapore_ships_chips_neg/chips0 /opt/data/datasets/detection/singapore_ships_chips_neg/chips0.csv \
    /opt/data/datasets/detection/singapore_ships_chips_neg/chips2 \ /opt/data/datasets/detection/singapore_ships_chips_neg/chips2.csv
```

When this is done, zip the `singapore_ships_chips_neg` folder into `singapore_ships_chips_neg.zip` and upload to S3 inside `s3://raster-vision/datasets/detection`. Also, separately upload the label map file to `s3://raster-vision/datasets/detection/ships_label_map.pbtxt`.

## Training a model on EC2

The training algorithm is configured in a file, an example of which can be seen at `configs/ships/ssd_mobilenet_v1.config`. This file needs to be modified if the local paths for data files change, or to tweak the hyperparameters or model architecture.
Once this file is committed to a branch (with name denoted by `branch-name`) and pushed to the remote repo, start a training job on AWS Batch, by running the following *from the VM*. Note the quotes around the command to run on the container.
```
src/detection/scripts/batch_submit.py  <branch-name> \
    "python -m rv.run train \
        --sync-interval 600 \
        /opt/src/detection/configs/ships/neg/ssd_mobilenet_v1.config \
        s3://raster-vision/datasets/detection/singapore_ships_chips_neg.zip \
        s3://raster-vision/datasets/detection/models/ssd_mobilenet_v1_coco_11_06_2017.zip \
        s3://raster-vision/results/detection/train/lhf_ships_neg"
    --attempts 1
```

This requires that there is a model checkpoint file (for using a pre-trained model) at `s3://raster-vision/datasets/detection/ssd_mobilenet_v1_coco_11_06_2017.zip`. This file can be downloaded from the website for the TF Object Detection API.
As training progresses, the checkpoints are periodically sync'd to S3 according to the `sync-interval` (which is in seconds).
You can monitor training using Tensorboard by pointing your browser at `http://<ec2 instance ip>:6006`. It may take a few minutes before results show up. When you are done training the model (ie. after the total loss flattens out), you need to kill the Batch job since it's running in an infinite loop. If this script fails due to instance shutdown and is run again, it should pick up where it left off using saved checkpoints.

## Making predictions

After training finishes, you can make predictions for a set of GeoTIFF files.  
First, in order to make the trained model available for inference, you must first convert a checkpoint file into an inference graph. For example, you can do the following.

```
aws s3 cp s3://raster-vision/results/detection/train/lhf_ships_neg0/train/model.ckpt-57662.index /tmp/
aws s3 cp s3://raster-vision/results/detection/train/lhf_ships_neg0/train/model.ckpt-57662.meta /tmp/
aws s3 cp s3://raster-vision/results/detection/train/lhf_ships_neg0/train/model.ckpt-57662.data-00000-of-00001 /tmp/

python models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /opt/src/detection/configs/ships/neg/ssd_mobilenet_v1.config \
    --checkpoint_path /tmp/model.ckpt-57662 \
    --inference_graph_path

aws s3 cp
/tmp/inference_graph.pb s3://raster-vision/results/detection/train/lhf_ships_neg0/train/
```

Next, upload some TIFF files to S3. If you would like to remove predictions that are contained inside a mask multi-polygon, you should also upload a GeoJSON file with the mask to S3. To run a prediction job locally, run the following.

```
python -m rv.run predict \
    --agg-predictions-debug-uri s3://raster-vision/results/detection/predict/lhf_ships3_crop/agg_predictions.png \
    s3://raster-vision/results/detection/train/lhf_ships_neg0/train/inference_graph.pb \
    s3://raster-vision/datasets/detection/ships_label_map.pbtxt \
    s3://raster-vision/results/detection/predict/lhf_ships3_crop/crop1.tif \
    s3://raster-vision/results/detection/predict/lhf_ships3_crop/crop2.tif \
    s3://raster-vision/results/detection/predict/lhf_ships3_crop/agg_predictions.json
```

When this is finished running, there should be a GeoJSON file with predictions at     `s3://raster-vision/results/detection/predict/lhf_ships3_crop/agg_predictions.json`. Note that you can use local paths instead of S3 URIs.

## Evaluating predictions

Aside from qualitatively evaluating the predictions in QGIS, you can quantify how good the predictions are compared to the ground truth using a script as follows. This outputs a JSON file with the precision and recall for each class.

```
python -m rv.run eval_predictions \
    /opt/data/datasets/detection/singapore_ships/test/1.tif \
    /opt/data/datasets/detection/singapore_ships/test/1.geojson \
    /opt/data/results/detection/predict/lhf_ships1/agg_predictions.json \
    /opt/data/datasets/detection/ships_label_map.pbtxt \
    /opt/data/results/detection/predict/lhf_ships1/eval.json
```

## Debugging

To debug, it may be helpful to run the above two scripts locally, and inspect the temporary files generated inside `/opt/data/temp`.
