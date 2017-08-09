#!/bin/bash

BRANCH=$1
COMMAND="${@:2}"

git clone -b $BRANCH https://github.com/azavea/raster-vision.git /tmp/raster-vision
cp -R /tmp/raster-vision/src/* /opt/src/
$COMMAND
