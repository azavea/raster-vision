variable "aws_key_name" {
  default = "open-tree-id"
}

variable "aws_vpc_id" {
  default = "vpc-3aa9ab5d"
}

variable "fleet_iam_role_arn" {
  default = "arn:aws:iam::002496907356:role/aws-ec2-spot-fleet-role"
}

variable "fleet_instance_profile" {
  default = "OpenTreeIDInstanceProfile"
}

variable "fleet_spot_price" {
  default = "0.9"
}

variable "fleet_allocation_strategy" {
  default = "lowestPrice"
}

variable "fleet_target_capacity" {
  default = "1"
}

variable "fleet_security_group_id" {
  default = "sg-0ed0cc74"
}

variable "fleet_ami" {
  default = "ami-50b4f047"
}
