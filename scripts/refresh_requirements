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

# re-build package wheels
echo "Rebuilding Raster Vision packages ..."
"$SCRIPTS_DIR"/build_packages >/dev/null 2>/dev/null
echo "Done"

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

pyproject_files=("pyproject.toml")
for plugin in "${plugins[@]}"; do
    pyproject_files+=("$plugin/pyproject.toml")
    exclusion_options+=(--no-emit-package "$plugin")
done

pushd "$SRC_DIR" >/dev/null

echo "Compiling requirements ..."
uv pip compile \
    --refresh --all-extras \
    "${pyproject_files[@]}" \
    "${exclusion_options[@]}" \
    --output-file "requirements.txt" >/dev/null
echo "Updated requirements.txt"

popd >/dev/null
