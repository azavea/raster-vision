ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y software-properties-common python-software-properties

RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && \
    apt-get install -y wget=1.* git=1:2.* python-protobuf=2.* python3-tk=3.* \
                       jq=1.5* \
                       build-essential libsqlite3-dev=3.11.* zlib1g-dev=1:1.2.* \
                       libopencv-dev=2.4.* python-opencv=2.4.* unzip curl && \
    apt-get autoremove && apt-get autoclean && apt-get clean

# See https://github.com/mapbox/rasterio/issues/1289
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install Python 3.6
RUN wget -q -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib/:$LD_LIBRARY_PATH
RUN conda install -y python=3.6
RUN python -m pip install --upgrade pip
RUN conda install -y -c conda-forge gdal=3.0.4

# Setup GDAL_DATA directory, rasterio needs it.
ENV GDAL_DATA=/opt/conda/lib/python3.6/site-packages/rasterio/gdal_data/

WORKDIR /opt/src/

COPY ./requirements-dev.txt /opt/src/requirements-dev.txt
RUN pip install -r requirements-dev.txt

# Ideally we'd just pip install each package, but if we do that, then a lot of the image
# will have to be re-built each time we make a change to source code. So, we split the
# install into installing all the requirements first (filtering out any prefixed with
# rastervision_*), and then copy over the source code.

# Install requirements for each package.
COPY ./rastervision_pipeline/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "rastervision_*" requirements.txt)

COPY ./rastervision_aws_s3/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "rastervision_*" requirements.txt)

COPY ./rastervision_aws_batch/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "rastervision_*" requirements.txt)

COPY ./rastervision_core/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "rastervision_*" requirements.txt)

COPY ./rastervision_pytorch_learner/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "rastervision_*" requirements.txt)

COPY ./rastervision_gdal_vsi/requirements.txt /opt/src/requirements.txt
RUN pip install $(grep -ivE "rastervision_*" requirements.txt)

# Commented out because there are no non-RV deps and it will fail if uncommented.
# COPY ./rastervision_pytorch_backend/requirements.txt /opt/src/requirements.txt
# RUN pip install $(grep -ivE "rastervision_*" requirements.txt)

# Install docs/requirements.txt
COPY ./docs/requirements.txt /opt/src/docs/requirements.txt
RUN pip install -r docs/requirements.txt

COPY scripts /opt/src/scripts/
COPY scripts/rastervision /usr/local/bin/rastervision
COPY tests /opt/src/tests/
COPY integration_tests /opt/src/integration_tests/
COPY .flake8 /opt/src/.flake8
COPY .coveragerc /opt/src/.coveragerc

# Needed for click to work
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
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

CMD ["bash"]
