#!/bin/bash -ex
# Install ecs-init, start docker and install nvidia-docker

NVIDIA_DOCKER_FULL_VERSION="1.0.1-1"
NVIDIA_DOCKER_VERSION="$(echo "${NVIDIA_DOCKER_FULL_VERSION}" | sed 's/\(.*\)-.*/\1/')"

# Nvidia GPU optimizations
# See http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html#optimize_gpu

# Check driver install status
nvidia-smi -q | head

# Make settings persistent
sudo nvidia-smi -pm 1

# Disable autoboost
sudo nvidia-smi --auto-boost-default=0

# Set GPU clock speeds
sudo nvidia-smi -ac 2505,875

sudo yum update
sudo yum install -y ecs-init
sudo service docker start
wget "https://github.com/NVIDIA/nvidia-docker/releases/download/v${NVIDIA_DOCKER_VERSION}/nvidia-docker-${NVIDIA_DOCKER_FULL_VERSION}.x86_64.rpm"
sudo rpm -ivh --nodeps "nvidia-docker-${NVIDIA_DOCKER_FULL_VERSION}.x86_64.rpm"


# Validate installation
rpm -ql nvidia-docker

# Make sure the NVIDIA kernel modules and driver files are bootstraped
# Otherwise running a GPU job inside a container will fail with "cuda: unknown exception"
echo '#!/bin/bash' | sudo tee /var/lib/cloud/scripts/per-boot/00_nvidia-modprobe > /dev/null
echo 'nvidia-modprobe -u -c=0' | sudo tee --append /var/lib/cloud/scripts/per-boot/00_nvidia-modprobe > /dev/null
sudo chmod +x /var/lib/cloud/scripts/per-boot/00_nvidia-modprobe
sudo /var/lib/cloud/scripts/per-boot/00_nvidia-modprobe

# Start the nvidia-docker-plugin and run a container with 
# nvidia-docker (retry up to 4 times if it fails initially)
sudo -b nohup nvidia-docker-plugin > /tmp/nvidia-docker.log
sudo docker pull nvidia/cuda
COMMAND="sudo nvidia-docker run nvidia/cuda nvidia-smi"
for i in {1..5}; do $COMMAND && break || sleep 15; done

# Create symlink to latest nvidia-driver version
nvidia_base=/var/lib/nvidia-docker/volumes/nvidia_driver
sudo ln -s $nvidia_base/$(ls $nvidia_base | sort -n  | tail -1) $nvidia_base/latest

sudo status ecs | grep running && sudo stop ecs
sudo rm -rf /var/lib/ecs/data/ecs_agent_data.json \
	"nvidia-docker-${NVIDIA_DOCKER_FULL_VERSION}.x86_64.rpm" \
	"NVIDIA-Linux-x86_64-${NVIDIA_DRIVERS_VERSION}.run"
