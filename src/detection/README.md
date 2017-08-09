### Overview
This is a demonstration of how to use the Tensorflow Object Detection API to train a detection model and then make predictions with it on AWS EC2. It uses the Pets dataset, but it should be easy to swap this out for the ships dataset using the `create_ships_tf_record.py` script. This code is very different than the rest of the Raster Vision codebase, so it's in its own directory which is `src/detection`. The TF API doesn't have a pip installer so I've just included it in the repo for now.

Here is a sample detection on the Pets dataset:
![samoyed_143](https://user-images.githubusercontent.com/1896461/29135882-8f155546-7d09-11e7-854c-c7152c590275.jpg)

### Running it
To start a training job, run the following from the VM
```
src/detection/scripts/batch_submit.py lf/detect \
    /opt/src/detection/scripts/train_ec2.sh \ configs/ssd_mobilenet_v1_pets.config pets0
```
You can view the progress of the training using Tensorboard by pointing your browser at `<ec2 instance ip>:6006`. When you are satisfied with the results, you need to kill the job since it's running in an infinite loop. Recent model checkpoints are synced to the S3 bucket under `results/detection/pet0`.

To start a prediction job, you can run
```
src/detection/scripts/batch_submit.py lf/detect \
    /opt/src/detection/scripts/predict_ec2.sh \
    /opt/src/detection/configs/ssd_mobilenet_v1_pets.config pets0 135656
```
which will put predictions in the S3 bucket in `results/detection/pets0/predictions`.

The real ships dataset isn't ready yet, so we are using a mock ships dataset. To convert this to TFRecord format, run this command locally in the CPU container.
```
python src/detection/scripts/create_ships_tf_record.py \
    --data_dir=/opt/data/datasets/detection/mock_ships \
    --output_dir=/opt/data/datasets/detection/mock_ships \
    --label_map_path=/opt/data/datasets/detection/mock_ships/ships_label_map.pbtxt
```
