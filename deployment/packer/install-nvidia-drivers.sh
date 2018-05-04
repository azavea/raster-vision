#!/bin/bash -ex

# This stuff is commented out because if it runs, it seems to successfully install the
# drivers, but then when the resulting AMI starts, running nvidia-smi reports that
# NVidia drivers aren't installed. The --install-libglvnd flag was added to make the
# install process complete without failing, but might be the reason things aren't working.
# The base AMI actually has the drivers we need, so this script isn't necessary, but it might
# be in the future.

# Update nvidia drivers
# NVIDIA_DRIVERS_VERSION="384.125"
# wget "http://us.download.nvidia.com/tesla/${NVIDIA_DRIVERS_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVERS_VERSION}.run"
# sudo /bin/bash "./NVIDIA-Linux-x86_64-${NVIDIA_DRIVERS_VERSION}.run" -sq --install-libglvnd
sudo reboot
