# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.require_version ">= 1.8"

$CHANGE_DIR = <<SCRIPT
  if ! grep -q "cd /vagrant" "/home/vagrant/.bashrc"; then
      echo "cd /vagrant" >> "/home/vagrant/.bashrc"
  fi
SCRIPT

DATA_DIR = ENV["RASTER_VISION_DATA_DIR"]
if DATA_DIR.nil?
  puts "ERROR: You must set the environment variable: RASTER_VISION_DATA_DIR"
  exit 1
end

NOTEBOOK_DIR = ENV["RASTER_VISION_NOTEBOOK_DIR"]
if NOTEBOOK_DIR.nil?
  puts "WARN: To use juptyer, You must set the environment variable: RASTER_VISION_NOTEBOOK_DIR"
end

S3_BUCKET = ENV.fetch("RASTER_VISION_BUCKET", "raster-vision")

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/trusty64"

  config.vm.define "raster_vision" do |raster_vision|
    raster_vision.vm.hostname = "raster-vision"
    raster_vision.vm.synced_folder "~/.aws", "/home/vagrant/.aws"
    raster_vision.vm.synced_folder DATA_DIR, "/opt/data"

    # If RASTER_VISION_NOTEBOOK_DIR is set, sync it to the VM
    if NOTEBOOK_DIR and File.directory?(File.expand_path(NOTEBOOK_DIR))
      raster_vision.vm.synced_folder NOTEBOOK_DIR, "/opt/notebooks"
    end

    raster_vision.vm.network "forwarded_port", guest: 8888, host: 8888

    raster_vision.vm.provider :virtualbox do |vb|
      vb.memory = 8192
      vb.cpus = 2
    end

    # Change working directory to /vagrant upon session start.
    raster_vision.vm.provision "shell", inline: $CHANGE_DIR

    if Vagrant::Util::Platform.windows?
      raster_vision.vm.provision "shell" do |shell|
        shell.binary = true
        shell.path = "scripts/provision.sh"
        shell.args = "deployment/ansible/raster-vision.yml deployment/ansible/roles.yml"
        shell.extra_vars = {
            s3_bucket: S3_BUCKET
        }
      end
    else
      raster_vision.vm.provision "ansible" do |ansible|
        ansible.playbook = "deployment/ansible/raster-vision.yml"
        ansible.galaxy_role_file = "deployment/ansible/roles.yml"
        ansible.extra_vars = {
            s3_bucket: S3_BUCKET
        }
      end
    end

    raster_vision.ssh.forward_agent = true
  end
end
