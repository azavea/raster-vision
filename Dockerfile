FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y software-properties-common python-software-properties

RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && \
    apt-get install -y wget=1.* git=1:2.* python-protobuf=2.* python3-tk=3.* \
                       gdal-bin=2.2.* \
                       jq=1.5* \
                       build-essential libsqlite3-dev=3.11.* zlib1g-dev=1:1.2.* \
                       libopencv-dev=2.4.* python-opencv=2.4.* unzip curl && \
    apt-get autoremove && apt-get autoclean && apt-get clean

# Setup GDAL_DATA directory, rasterio needs it.
ENV GDAL_DATA=/usr/share/gdal/2.2/

# See https://github.com/mapbox/rasterio/issues/1289
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install protoc
RUN wget -O /tmp/protoc3.zip https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip && \
    unzip /tmp/protoc3.zip -d /tmp/protoc3 && \
    mv /tmp/protoc3/bin/* /usr/local/bin/ && \
    mv /tmp/protoc3/include/* /usr/local/include/ && \
    rm -R /tmp/protoc3 && \
    rm /tmp/protoc3.zip

# Install Python 3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=3.6

# Install core deep learning libs
RUN pip install tensorflow-gpu==1.14.*
RUN pip install keras==2.2.*
RUN pip install torch==1.1.* torchvision==0.3.*
RUN pip install fastai==1.0.57

# Set WORKDIR and PYTHONPATH
WORKDIR /opt/src/
ENV PYTHONPATH=/opt/src:/opt/tf-models:/opt/tf-models/slim:$PYTHONPATH

# Install TF Object Detection API and Deeplab in /opt/tf-models
RUN mkdir -p /opt/tf-models/temp/ && \
    cd /opt/tf-models/temp/ && \
    git clone --single-branch -b AZ-v1.11-RV-v0.8.0 https://github.com/azavea/models.git && \
    mv models/research/object_detection/ ../object_detection && \
    mv models/research/deeplab/ ../deeplab && \
    mv models/research/slim/ ../slim && \
    cd .. && \
    rm -R temp && \
    protoc object_detection/protos/*.proto --python_out=. && \
    pip install cython==0.28.* && \
    pip install pycocotools==2.0.*

# Install Tippecanoe
RUN cd /tmp && \
    wget https://github.com/mapbox/tippecanoe/archive/1.32.5.zip && \
    unzip 1.32.5.zip && \
    cd tippecanoe-1.32.5 && \
    make && \
    make install

# Install requirements-dev.txt
COPY ./requirements-dev.txt /opt/src/requirements-dev.txt
RUN pip install -r requirements-dev.txt

# Install docs/requirements.txt
COPY ./docs/requirements.txt /opt/src/docs/requirements.txt
RUN pip install -r docs/requirements.txt

# Install extras_requirements.json
# Don't install tensorflow
COPY ./extras_requirements.json /opt/src/extras_requirements.json
RUN cat extras_requirements.json | jq  '.[][]' | grep -v 'tensorflow' | sort -u | xargs pip install

# Install requirements.txt
COPY ./requirements.txt /opt/src/requirements.txt
RUN pip install -r requirements.txt

# Install optional-requirements.txt
COPY ./optional-requirements.txt /opt/src/optional-requirements.txt
RUN pip install -r optional-requirements.txt

# Needed for click to work
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

COPY scripts/rastervision /usr/local/bin/
COPY rastervision /opt/src/rastervision
COPY scripts/compile /opt/src/scripts/compile
RUN /opt/src/scripts/compile