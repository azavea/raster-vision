#!/bin/bash

########################################################################
# Compiles requirements from pyproject.toml into requirements.txt files.
########################################################################

set -euo pipefail

# Determine the script's directory (even if it's a symbolic link)
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do SOURCE="$(readlink "$SOURCE")"; done
SCRIPTS_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
SRC_DIR="$(cd -P "$(dirname "$SCRIPTS_DIR")" && pwd)"

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

# compile requirements
for dir in "$SRC_DIR/${plugins[@]}"; do
    echo "Processing $dir"
    uv pip compile \
        --refresh --all-extras \
        "$dir/pyproject.toml" \
        --output-file "$dir/requirements.txt"
done

uv pip compile \
    --refresh --all-extras \
    "$SRC_DIR/pyproject.toml" \
    --output-file "$SRC_DIR/requirements.txt"
