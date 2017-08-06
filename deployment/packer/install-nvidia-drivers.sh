#!/bin/bash -ex

# Update nvidia drivers
NVIDIA_DRIVERS_VERSION="375.51"

wget "http://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVERS_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVERS_VERSION}.run"
sudo /bin/bash "./NVIDIA-Linux-x86_64-${NVIDIA_DRIVERS_VERSION}.run" -asq
sudo reboot

