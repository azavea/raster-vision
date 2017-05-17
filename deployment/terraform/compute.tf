#
# Spot Fleet Resources
#

data "template_file" "init" {
  template = "${file("config-ec2")}"

  vars {
    s3_bucket = "${var.s3_bucket}"
  }
}

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
    subnet_id                   = "subnet-b2f915e8"
    vpc_security_group_ids      = ["${var.fleet_security_group_id}"]
    user_data                   = "${data.template_file.init.rendered}"

    root_block_device {
      volume_size = "128"
      volume_type = "gp2"
    }
  }

  launch_specification {
    iam_instance_profile        = "${var.fleet_instance_profile}"
    instance_type               = "p2.xlarge"
    ami                         = "${var.fleet_ami}"
    key_name                    = "${var.aws_key_name}"
    subnet_id                   = "subnet-7f16321a"
    vpc_security_group_ids      = ["${var.fleet_security_group_id}"]
    user_data                   = "${data.template_file.init.rendered}"

    root_block_device {
      volume_size = "128"
      volume_type = "gp2"
    }
  }

  launch_specification {
    iam_instance_profile        = "${var.fleet_instance_profile}"
    instance_type               = "p2.xlarge"
    ami                         = "${var.fleet_ami}"
    key_name                    = "${var.aws_key_name}"
    subnet_id                   = "subnet-e121d7cd"
    vpc_security_group_ids      = ["${var.fleet_security_group_id}"]
    user_data                   = "${data.template_file.init.rendered}"

    root_block_device {
      volume_size = "128"
      volume_type = "gp2"
    }
  }

  launch_specification {
    iam_instance_profile        = "${var.fleet_instance_profile}"
    instance_type               = "p2.xlarge"
    ami                         = "${var.fleet_ami}"
    key_name                    = "${var.aws_key_name}"
    subnet_id                   = "subnet-173c9c5f"
    vpc_security_group_ids      = ["${var.fleet_security_group_id}"]
    user_data                   = "${data.template_file.init.rendered}"

    root_block_device {
      volume_size = "128"
      volume_type = "gp2"
    }
  }
}
