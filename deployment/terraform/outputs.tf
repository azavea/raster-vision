output "fleet_request_id" {
  value = "${aws_spot_fleet_request.gpu_worker.id}"
}

output "fleet_request_state" {
  value = "${aws_spot_fleet_request.gpu_worker.spot_request_state}"
}
