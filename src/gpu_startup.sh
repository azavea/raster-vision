#!/bin/bash
# Modified from https://blog.cloudsight.ai/deep-learning-image-recognition-using-gpus-in-amazon-ecs-docker-containers-5bdb1956f30e

set -e

echo "Copying the NVidia drivers from the parent..."
find /hostusr -name "*nvidia*" -o -name "*cuda*" -o -name "*GL*" | while read path
do
  newpath="/usr${path#/hostusr}"
  mkdir -p `dirname $newpath` && \
    cp -a $path $newpath
done

cp -ar /hostlib/modules /lib

echo "/usr/lib64" > /etc/ld.so.conf.d/nvidia.conf
ldconfig

set -x
$@
