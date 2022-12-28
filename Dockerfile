ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

# wget: needed below to install conda
# build-essential: installs gcc which is needed to install some deps like rasterio
# libGL1: needed to avoid following error when using cv2
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# See https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && \
    apt-get install -y wget=1.* build-essential libgl1 curl && \
    apt-get autoremove && apt-get autoclean && apt-get clean

ARG PYTHON_VERSION=3.9
ARG TARGETPLATFORM

RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  LINUX_ARCH=aarch64  ;; \
         *)              LINUX_ARCH=x86_64   ;; \
    esac && echo ${LINUX_ARCH} > /root/linux_arch

# needed for jupyter lab extensions
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs

# Install Python and conda
RUN wget -q -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(cat /root/linux_arch).sh && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib/:$LD_LIBRARY_PATH
RUN conda install -y python=${PYTHON_VERSION}
RUN python -m pip install --upgrade pip

# We need to install GDAL first to install Rasterio on non-AMD64 architectures.
# The Rasterio wheels contain GDAL in them, but they are only built for AMD64 now.
RUN conda install -y -c conda-forge gdal=3.5.2
ENV GDAL_DATA=/opt/conda/lib/python${PYTHON_VERSION}/site-packages/rasterio/gdal_data/

# This is to prevent the following error when starting the container.
# bash: /opt/conda/lib/libtinfo.so.6: no version information available (required by bash)
# See https://askubuntu.com/questions/1354890/what-am-i-doing-wrong-in-conda
RUN rm /opt/conda/lib/libtinfo.so.6 && \
    ln -s /lib/$(cat /root/linux_arch)-linux-gnu/libtinfo.so.6 /opt/conda/lib/libtinfo.so.6

WORKDIR /opt/src/

COPY ./requirements-dev.txt /opt/src/requirements-dev.txt
RUN pip install -r requirements-dev.txt

# Ideally we'd just pip install each package, but if we do that, then a lot of the image
# will have to be re-built each time we make a change to source code. So, we split the
# install into installing all the requirements first (filtering out any prefixed with
# rastervision_*), and then copy over the source code.

# Install requirements for each package.
# -E "^\s*$|^#|rastervision_*" means exclude blank lines, comment lines,
# and rastervision plugins.
COPY ./rastervision_pipeline/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "^\s*$|^#|rastervision_*" requirements.txt)

COPY ./rastervision_aws_s3/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "^\s*$|^#|rastervision_*" requirements.txt)

COPY ./rastervision_aws_batch/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "^\s*$|^#|rastervision_*" requirements.txt)

COPY ./rastervision_core/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "^\s*$|^#|rastervision_*" requirements.txt)

COPY ./rastervision_pytorch_learner/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "^\s*$|^#|rastervision_*" requirements.txt)

COPY ./rastervision_gdal_vsi/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "^\s*$|^#|rastervision_*" requirements.txt)

# Commented out because there are no non-RV deps and it will fail if uncommented.
# COPY ./rastervision_pytorch_backend/requirements.txt /opt/src/requirements.txt
# RUN pip install $(grep -ivE "^\s*$|^#|rastervision_*" requirements.txt)

#########################
# Docs
#########################
# Install docs/requirements.txt
COPY ./docs/requirements.txt /opt/src/docs/requirements.txt
RUN pip install -r docs/requirements.txt

# Install pandoc, needed for rendering notebooks
# Get latest release link from here: https://github.com/jgm/pandoc/releases
ARG TARGETARCH
RUN wget https://github.com/jgm/pandoc/releases/download/2.19.2/pandoc-2.19.2-1-${TARGETARCH}.deb && \
    dpkg -i pandoc-2.19.2-1-${TARGETARCH}.deb && rm pandoc-2.19.2-1-${TARGETARCH}.deb
#########################

# Fix CI problem by pinning pyopenssl version
# See https://askubuntu.com/questions/1428181/module-lib-has-no-attribute-x509-v-flag-cb-issuer-check
RUN python -m pip install --upgrade pyopenssl==22.0.0

COPY scripts /opt/src/scripts/
COPY scripts/rastervision /usr/local/bin/rastervision
COPY tests /opt/src/tests/
COPY integration_tests /opt/src/integration_tests/
COPY .flake8 /opt/src/.flake8
COPY .coveragerc /opt/src/.coveragerc

# Needed for click to work
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
# Needed for GDAL 3.0
ENV PROJ_LIB /opt/conda/share/proj/

# Copy code for each package.
ENV PYTHONPATH=/opt/src:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_pipeline/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_aws_s3/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_aws_batch/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_gdal_vsi/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_core/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_pytorch_learner/:$PYTHONPATH
ENV PYTHONPATH=/opt/src/rastervision_pytorch_backend/:$PYTHONPATH

COPY ./rastervision_pipeline/ /opt/src/rastervision_pipeline/
COPY ./rastervision_aws_s3/ /opt/src/rastervision_aws_s3/
COPY ./rastervision_aws_batch/ /opt/src/rastervision_aws_batch/
COPY ./rastervision_core/ /opt/src/rastervision_core/
COPY ./rastervision_pytorch_learner/ /opt/src/rastervision_pytorch_learner/
COPY ./rastervision_pytorch_backend/ /opt/src/rastervision_pytorch_backend/
COPY ./rastervision_gdal_vsi/ /opt/src/rastervision_gdal_vsi/

# This gets rid of the following error when importing cv2 on arm64.
# We cannot use the ENV directive since it cannot be used conditionally.
# See https://github.com/opencv/opencv/issues/14884
# ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
RUN if [${TARGETARCH} == "arm64"]; \
    then echo "export LD_PRELOAD=/lib/$(cat /root/linux_arch)-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD" >> /root/.bashrc; fi

CMD ["bash"]
