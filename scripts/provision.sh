#!/bin/bash

# This script supports provisioning on Windows machines.
# Inspired by https://github.com/ontic/ansible-windows/blob/master/provision.sh
#
# This script expects at least 2 arguments:
#
# $ provision.sh "ansible/playboook.yml ansible/roles.yml foo=bar"
#
# 1 - Path to the Ansible playbook file
# 2 - Path to the Ansible roles file
# 3 - [Optional] Extra variables passed to Ansible

PLAYBOOK_FILE=$1
ROLES_FILE=$2
EXTRA_VARS=$3
PLAYBOOK_DIR=${PLAYBOOK_FILE%/*}

IS_DEBIAN=$(which apt-get 2>/dev/null)

if [ ! -f "/vagrant/$PLAYBOOK_FILE" ]; then
	echo "Cannot find Ansible playbook file."
	exit 1
fi

if [ ! -f "/vagrant/$ROLES_FILE" ]; then
	echo "Cannot find Ansible roles file"
	exit 1
fi

if ! command -v ansible >/dev/null; then
	echo "Installing Ansible dependencies and Git."
	if [ ! -z "$IS_DEBIAN" ]; then
		apt-get update -y
		apt-get install -y git python python-dev python-pip build-essential
	else
		echo "Your operating system is not supported."
		exit 1
	fi

	echo "Installing required python modules and Ansible."
	pip install paramiko pyyaml markupsafe jinja2 ansible
fi

echo "Installing Ansible roles from requirements file, if available."
find "/vagrant/$PLAYBOOK_DIR" \( -name "requirements.yml" -o -name "requirements.txt" \) -exec sudo ansible-galaxy install -r {} \;

if [ -z "$EXTRA_VARS" ]; then
	ansible-galaxy install -r "/vagrant/${ROLES_FILE}"
	ansible-playbook "/vagrant/${PLAYBOOK_FILE}" --connection=local
else
	ansible-galaxy install -r "/vagrant/${ROLES_FILE}"
	ansible-playbook "/vagrant/${PLAYBOOK_FILE}" --extra-vars ${EXTRA_VARS} --connection=local
fi
