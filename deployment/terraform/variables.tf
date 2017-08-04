variable "aws_key_name" {
  default = "raster-vision"
}

variable "aws_vpc_id" {
  default = "vpc-7e3f2618"
}

variable "fleet_iam_role_arn" {
  default = "arn:aws:iam::279682201306:role/aws-ec2-spot-fleet-role"
}

variable "fleet_instance_profile" {
  default = "raster-vision-instance-profile"
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
  default = "sg-4c00d332"
}

variable "fleet_ami" {
  default = "ami-3583a74e"
}

variable "s3_bucket" {
  default = "raster-vision"
}
