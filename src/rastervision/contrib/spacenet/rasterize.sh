#!/usr/bin/env bash

aoi=$1
targets=$2
src=$3
sink=$4
tempfile=$(mktemp /tmp/XXXX.tif)

set -x
gdal_merge.py -createonly -init "0 255 0" -co "COMPRESS=DEFLATE" -o $tempfile $src
gdal_rasterize -b 1 -b 2 -b 3 -burn 0 -burn 0 -burn 255 $targets $tempfile # target class
gdal_rasterize -i -b 1 -b 2 -b 3 -burn 0 -burn 0 -burn 0 $aoi $tempfile # AOI mask
cp $tempfile $sink
rm $tempfile
