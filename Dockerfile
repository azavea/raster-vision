ARG BUILD_TYPE
ARG CUDA_VERSION
ARG UBUNTU_VERSION

########################################################################

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} as thinbuild

ARG PYTHON_VERSION=3.11

# build-essential: installs gcc which is needed to install some deps like rasterio
# libGL1: needed to avoid following error when using cv2
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# See https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN --mount=type=cache,target=/var/cache/apt apt update && \
    apt install -y wget=1.21.2-2ubuntu1 build-essential=12.9ubuntu3 libgl1=1.4.0-1 curl=7.81.0-1ubuntu1.13 git=1:2.34.1-1ubuntu1.10 tree=2.0.2-1 gdal-bin=3.4.1+dfsg-1build4 libgdal-dev=3.4.1+dfsg-1build4 python${PYTHON_VERSION} python3-pip && \
    curl -fsSL https://deb.nodesource.com/node_16.x | bash - && \
    apt install -y nodejs=16.20.2-deb-1nodesource1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    apt autoremove

########################################################################

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} as fullbuild

ARG TARGETPLATFORM
ARG PYTHON_VERSION=3.11

# wget: needed below to install conda
# build-essential: installs gcc which is needed to install some deps like rasterio
# libGL1: needed to avoid following error when using cv2
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# See https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && \
    apt-get install -y wget=1.* build-essential libgl1 curl git tree && \
    apt-get autoremove && apt-get autoclean && apt-get clean

RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  LINUX_ARCH=aarch64  ;; \
         *)              LINUX_ARCH=x86_64   ;; \
    esac && echo ${LINUX_ARCH} > /root/linux_arch

# needed for jupyter lab extensions
RUN curl -fsSL https://deb.nodesource.com/node_16.x | bash - && \
    apt-get install -y nodejs

# Install Python and conda/mamba (mamba installs conda as well)
RUN wget -q -O ~/micromamba.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(cat /root/linux_arch).sh && \
    chmod +x ~/micromamba.sh && \
    bash ~/micromamba.sh -b -p /opt/conda && \
    rm ~/micromamba.sh
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib/:$LD_LIBRARY_PATH
RUN mamba init
RUN mamba install -y python=${PYTHON_VERSION}
RUN python -m pip install --upgrade pip

# env variable required by uv
ENV CONDA_PREFIX=/opt/conda
RUN pip install uv

# We need to install GDAL first to install Rasterio on non-AMD64 architectures.
# The Rasterio wheels contain GDAL in them, but they are only built for AMD64 now.
RUN mamba update mamba -y && mamba install -y -c conda-forge gdal=3.6.3
ENV GDAL_DATA=/opt/conda/lib/python${PYTHON_VERSION}/site-packages/rasterio/gdal_data/
# Needed for GDAL 3.0
ENV PROJ_LIB /opt/conda/share/proj/

# This is to prevent the following error when starting the container.
# bash: /opt/conda/lib/libtinfo.so.6: no version information available (required by bash)
# See https://askubuntu.com/questions/1354890/what-am-i-doing-wrong-in-conda
RUN rm /opt/conda/lib/libtinfo.so.6 && \
    ln -s /lib/$(cat /root/linux_arch)-linux-gnu/libtinfo.so.6 /opt/conda/lib/libtinfo.so.6

# This gets rid of the following error when importing cv2 on arm64.
# We cannot use the ENV directive since it cannot be used conditionally.
# See https://github.com/opencv/opencv/issues/14884
# ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
RUN if [ "${TARGETARCH}" = "arm64" ]; \
    then echo "export LD_PRELOAD=/lib/$(cat /root/linux_arch)-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD" >> /root/.bashrc; fi

########################################################################

FROM ${BUILD_TYPE:-fullbuild} AS final_stage

ARG TARGETARCH

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /opt/src/

#------------------------------------------------------------------------

# Ideally we'd just pip install each package, but if we do that, then
# a lot of the image will have to be re-built each time we make a
# change to the code. So, we split the install into installing all the
# requirements in bunches (filtering out any prefixed with
# rastervision_*), and then copy over the source code.  The
# dependencies are installed in bunches rather than package-by-package
# or on a per-RV component basis to reduce the build time, the number
# of layers, and the overall image size, and to reduce churn
# (installing and uninstalling of Python packages during the build).
#
# The bunches are heuristic and are meant to keep the heaviest and/or
# least-frequently-changing dependencies before the more variable
# ones.  At time of writing, the amount of image size attributable to
# PyTorch (and the amount of image size overall) is heavily dominated
# by PyTorch, so it is first.

# Install requirements.
# -E "^\s*$|^#|rastervision_*" means exclude blank lines, comment lines,
# and rastervision plugins.

COPY ./rastervision_pytorch_learner/requirements.txt /opt/src/pytorch-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip cat pytorch-requirements.txt | sort | uniq > all-requirements.txt && \
    uv pip install $(grep -ivE "^\s*$|^#|rastervision_*" all-requirements.txt) && \
    rm all-requirements.txt

COPY ./rastervision_aws_batch/requirements.txt /opt/src/batch-requirements.txt
COPY ./rastervision_aws_s3/requirements.txt /opt/src/s3-requirements.txt
COPY ./rastervision_core/requirements.txt /opt/src/core-requirements.txt

# Pip wheels for triangle are missing for ARM64 architectures and building
# from source fails, so we skip it.
RUN if [ "${TARGETARCH}" = "arm64" ]; \
    then sed -i '/^triangle.*$/d' /opt/src/core-requirements.txt; fi

COPY ./rastervision_gdal_vsi/requirements.txt /opt/src/gdal-requirements.txt
COPY ./rastervision_pipeline/requirements.txt /opt/src/pipeline-requirements.txt
COPY ./rastervision_aws_sagemaker/requirements.txt /opt/src/sagemaker-requirements.txt
COPY ./requirements-dev.txt /opt/src/requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip cat \
    /opt/src/batch-requirements.txt \
    /opt/src/s3-requirements.txt \
    /opt/src/core-requirements.txt \
    /opt/src/gdal-requirements.txt \
    /opt/src/pipeline-requirements.txt \
    /opt/src/sagemaker-requirements.txt \
    /opt/src/requirements-dev.txt \
    | sort | uniq > all-requirements.txt && \
    uv pip install $(grep -ivE "^\s*$|^#|rastervision_*" all-requirements.txt) && \
    rm all-requirements.txt

#########################
# Docs
#########################
# Install docs/requirements.txt
COPY ./docs/requirements.txt /opt/src/docs/pandoc-requirements.txt

# Install pandoc, needed for rendering notebooks
# Get latest release link from here: https://github.com/jgm/pandoc/releases
RUN --mount=type=cache,target=/root/.cache/pip uv pip install -r docs/pandoc-requirements.txt && \
    wget https://github.com/jgm/pandoc/releases/download/3.1.12.2/pandoc-3.1.12.2-1-${TARGETARCH}.deb && \
    dpkg -i pandoc-3.1.12.2-1-${TARGETARCH}.deb && rm pandoc-3.1.12.2-1-${TARGETARCH}.deb

#------------------------------------------------------------------------

ENV PYTHONPATH=/opt/src:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_aws_batch/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_aws_s3/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_core/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_gdal_vsi/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_pipeline/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_aws_sagemaker/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_pytorch_backend/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_pytorch_learner/:$PYTHONPATH

COPY scripts /opt/src/scripts/
COPY scripts/rastervision /usr/local/bin/rastervision
COPY tests /opt/src/tests/
COPY integration_tests /opt/src/integration_tests/
COPY .flake8 /opt/src/.flake8
COPY .coveragerc /opt/src/.coveragerc

COPY ./rastervision_aws_batch/ /opt/src/rastervision_aws_batch/
COPY ./rastervision_aws_s3/ /opt/src/rastervision_aws_s3/
COPY ./rastervision_core/ /opt/src/rastervision_core/
COPY ./rastervision_gdal_vsi/ /opt/src/rastervision_gdal_vsi/
COPY ./rastervision_pipeline/ /opt/src/rastervision_pipeline/
COPY ./rastervision_aws_sagemaker/ /opt/src/rastervision_aws_sagemaker/
COPY ./rastervision_pytorch_backend/ /opt/src/rastervision_pytorch_backend/
COPY ./rastervision_pytorch_learner/ /opt/src/rastervision_pytorch_learner/

# needed for this image to be used by the AWS SageMaker PyTorch Estimator
RUN uv pip install sagemaker_pytorch_training==2.8.1
ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main

# Install a onnxruntime-gpu version compatible with CUDA 12. Specifying
# --extra-index-url in requirements.txt seems to cause problems with the
# RTD build.
RUN if [ "${TARGETARCH}" != "arm64" ]; then \
    uv pip install --upgrade onnxruntime-gpu==1.17 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/; fi

CMD ["bash"]
