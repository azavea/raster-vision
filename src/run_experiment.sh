#!/bin/bash

# Copies the git branch into the right place inside the container and then runs
# Raster Vision.

export S3_BUCKET=$1
branch=$2
run_args=${@:3}
git clone -b $branch https://github.com/azavea/raster-vision.git /tmp/raster-vision
cp -R /tmp/raster-vision/src/* /opt/src/
python -m rastervision.run $run_args
