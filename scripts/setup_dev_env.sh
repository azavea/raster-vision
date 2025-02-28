#!/bin/bash

########################################################################
# Install namespace package and all sub-packages in editable mode.
########################################################################

set -euo pipefail

# Determine the script's directory (even if it's a symbolic link)
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do SOURCE="$(readlink "$SOURCE")"; done
SCRIPTS_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
SRC_DIR="$(cd -P "$(dirname "$SCRIPTS_DIR")" && pwd)"

pip install --upgrade uv

plugins=(
    "rastervision_pipeline"
    "rastervision_aws_batch"
    "rastervision_aws_s3"
    "rastervision_core"
    "rastervision_pytorch_learner"
    "rastervision_pytorch_backend"
    "rastervision_gdal_vsi"
    "rastervision_aws_sagemaker"
)

uv pip sync "$SRC_DIR/requirements.txt"

for dir in "${plugins[@]}"; do
    uv pip install -e "$SRC_DIR/$dir" --no-deps
done

uv pip install -e "$SRC_DIR" --no-deps
