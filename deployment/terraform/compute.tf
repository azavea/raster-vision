#
# Spot Fleet Resources
#

resource "aws_spot_fleet_request" "gpu_worker" {
  iam_fleet_role                      = "${var.fleet_iam_role_arn}"
  spot_price                          = "${var.fleet_spot_price}"
  allocation_strategy                 = "${var.fleet_allocation_strategy}"
  terminate_instances_with_expiration = false
  excess_capacity_termination_policy  = "Default"
  target_capacity                     = "${var.fleet_target_capacity}"


  launch_specification {
    iam_instance_profile        = "${var.fleet_instance_profile}"
    instance_type               = "p2.xlarge"
    ami                         = "${var.fleet_ami}"
    key_name                    = "${var.aws_key_name}"
    subnet_id                   = "subnet-7aeb2121"
    vpc_security_group_ids      = ["${var.fleet_security_group_id}"]
    user_data                   = "${file("config-ec2")}"

    root_block_device {
      volume_size = "64"
      volume_type = "gp2"
    }
  }

  launch_specification {
    iam_instance_profile        = "${var.fleet_instance_profile}"
    instance_type               = "p2.xlarge"
    ami                         = "${var.fleet_ami}"
    key_name                    = "${var.aws_key_name}"
    subnet_id                   = "subnet-76d6134a"
    vpc_security_group_ids      = ["${var.fleet_security_group_id}"]
    user_data                   = "${file("config-ec2")}"

    root_block_device {
      volume_size = "64"
      volume_type = "gp2"
    }
  }

  launch_specification {
    iam_instance_profile        = "${var.fleet_instance_profile}"
    instance_type               = "p2.xlarge"
    ami                         = "${var.fleet_ami}"
    key_name                    = "${var.aws_key_name}"
    subnet_id                   = "subnet-42b83e0b"
    vpc_security_group_ids      = ["${var.fleet_security_group_id}"]
    user_data                   = "${file("config-ec2")}"

    root_block_device {
      volume_size = "64"
      volume_type = "gp2"
    }
  }

  launch_specification {
    iam_instance_profile        = "${var.fleet_instance_profile}"
    instance_type               = "p2.xlarge"
    ami                         = "${var.fleet_ami}"
    key_name                    = "${var.aws_key_name}"
    subnet_id                   = "subnet-5f538472"
    vpc_security_group_ids      = ["${var.fleet_security_group_id}"]
    user_data                   = "${file("config-ec2")}"

    root_block_device {
      volume_size = "64"
      volume_type = "gp2"
    }
  }
}
