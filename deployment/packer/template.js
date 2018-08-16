{
    "variables": {
        "raster_vision_gpu_version": "",
        "aws_region": "",
        "aws_gpu_ami": "",
        "branch": ""
    },
    "builders": [
        {
            "name": "raster-vision-gpu",
            "type": "amazon-ebs",
            "region": "{{user `aws_region`}}",
            "source_ami": "{{user `aws_gpu_ami`}}",
            "instance_type": "p3.2xlarge",
            "ssh_username": "ec2-user",
            "ami_name": "raster-vision-gpu-{{timestamp}}-{{user `branch`}}",
            "run_tags": {
                "PackerBuilder": "amazon-ebs"
            },
             "launch_block_device_mappings": [
            {
              "device_name": "/dev/xvda",
              "volume_size": 120,
              "volume_type": "gp2",
              "delete_on_termination": true
            },
            {
              "device_name": "/dev/xvdcz",
              "volume_size": 22,
              "volume_type": "gp2",
              "delete_on_termination": true
            }
          ],
            "tags": {
                "Name": "raster-vision-gpu",
                "Version": "{{user `raster_vision_gpu_version`}}",
                "Created": "{{ isotime }}"
            },
            "associate_public_ip_address": true
        }
    ],
    "provisioners": [
        {
            "type": "shell",
            "script": "./packer/scripts/configure-gpu.sh"
        },
        {
            "type": "shell",
            "script": "./packer/scripts/reset-ecs-state.sh"
        }
    ]
}
