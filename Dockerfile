FROM tensorflow/tensorflow:1.10.1-gpu-py3

# Everything below this line is common between CPU and GPU images.

# Needed for click to work
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Packages
RUN apt-get update && \
    apt-get install -y wget=1.* git=1:2.* python-protobuf=2.* python3-tk=3.*

# Install protoc
RUN wget -O /tmp/protoc3.zip https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip && \
    unzip /tmp/protoc3.zip -d /tmp/protoc3 && \
    mv /tmp/protoc3/bin/* /usr/local/bin/ && \
    mv /tmp/protoc3/include/* /usr/local/include/ && \
    rm -R /tmp/protoc3 && \
    rm /tmp/protoc3.zip

# Install GDAL
RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && \
    apt-get install -y gdal-bin=2.1.*

# Install jq for parsing extras-requirements.json
RUN apt-get install jq -y

# Install TF Object Detection API in /opt/tf-models
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

# Setup GDAL_DATA directory, rasterio needs it.
ENV GDAL_DATA=/usr/share/gdal/2.1/

# See https://github.com/mapbox/rasterio/issues/1289
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Make Python3 default Python
RUN rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

# Install tippecanoe
RUN apt-get -y upgrade \
  && apt-get -y install build-essential libsqlite3-dev zlib1g-dev

RUN cd /tmp && \
    git clone https://github.com/mapbox/tippecanoe.git && \
    cd tippecanoe && \
    make && \
    make install

# Set WORKDIR and PYTHONPATH
WORKDIR /opt/src/
ENV PYTHONPATH=/opt/src:/opt/tf-models:/opt/tf-models/slim:$PYTHONPATH

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
