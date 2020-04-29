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

# Install protoc
RUN wget -q -O /tmp/protoc3.zip https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip && \
    unzip /tmp/protoc3.zip -d /tmp/protoc3 && \
    mv /tmp/protoc3/bin/* /usr/local/bin/ && \
    mv /tmp/protoc3/include/* /usr/local/include/ && \
    rm -R /tmp/protoc3 && \
    rm /tmp/protoc3.zip

# Install Python 3.6
RUN wget -q -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=3.6 && conda install -c conda-forge -y gdal=3.0.4

# Setup GDAL_DATA directory, rasterio needs it.
ENV GDAL_DATA=/opt/conda/lib/python3.6/site-packages/rasterio/gdal_data/

WORKDIR /opt/src/
ENV PYTHONPATH=/opt/src:$PYTHONPATH

# Install Tippecanoe
RUN cd /tmp && \
    wget -q https://github.com/mapbox/tippecanoe/archive/1.32.5.zip && \
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

# Install extra requirements that apply to both types of images
COPY ./extras_requirements.json /opt/src/extras_requirements.json
RUN cat extras_requirements.json | jq  '.["aws"][]' | xargs pip install

# RUN cat extras_requirements.json | jq  '.["feature-extraction"][]' | xargs pip install
# TODO before release, we need to upgrade version of mask-to-polygons on pypi and
# update extra_requirements.json
RUN pip install git+git://github.com/azavea/mask-to-polygons@f1d0b623c648ba7ccb1839f74201c2b57229b006

# Install requirements.txt
COPY ./requirements.txt /opt/src/requirements.txt
RUN pip install -r requirements.txt

# Install optional-requirements.txt
COPY ./optional-requirements.txt /opt/src/optional-requirements.txt
RUN pip install -r optional-requirements.txt

# Needed for click to work
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PROJ_LIB /opt/conda/share/proj/
