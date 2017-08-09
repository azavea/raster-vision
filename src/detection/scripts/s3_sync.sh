#!/bin/sh
# Defining a timeout of 0.5 hour
TIMEOUT=$1
RUN=$2

while true
do
    aws s3 sync /opt/data/results/detection/$RUN s3://raster-vision/results/detection/$RUN --delete
    sleep $TIMEOUT
done
