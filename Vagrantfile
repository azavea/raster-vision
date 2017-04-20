# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.require_version ">= 1.8"

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/trusty64"

  config.vm.define "otid" do |otid|

  # Expose quiver
  config.vm.network "forwarded_port", { guest: 5000, host: 5000 }
    otid.vm.hostname = "otid"
    otid.vm.synced_folder "~/.aws", "/home/vagrant/.aws"
    otid.vm.synced_folder "~/data", "/opt/data"

    otid.vm.provider :virtualbox do |vb|
      vb.memory = 8192
      vb.cpus = 2
    end

    # Change working directory to /vagrant upon session start.
    otid.vm.provision "shell", inline: <<SCRIPT
      if ! grep -q "cd /vagrant" "/home/vagrant/.bashrc"; then
          echo "cd /vagrant" >> "/home/vagrant/.bashrc"
      fi
SCRIPT

    otid.vm.provision "ansible" do |ansible|
      ansible.playbook = "deployment/ansible/open-tree-id.yml"
      ansible.galaxy_role_file = "deployment/ansible/roles.yml"
    end

    otid.ssh.forward_agent = true
  end
end
