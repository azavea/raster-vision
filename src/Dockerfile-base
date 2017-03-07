# Based on https://github.com/fchollet/keras/blob/2b51317be82d4420169d2cc79dc4443028417911/docker/Dockerfile
FROM nvidia/cuda:7.5-cudnn5-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-3.9.1-Linux-x86_64.sh && \
    echo "6c6b44acdd0bc4229377ee10d52c8ac6160c336d9cdd669db7371aa9344e1ac3 *Miniconda3-3.9.1-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-3.9.1-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-3.9.1-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

ENV PYTHONPATH='/src/:$PYTHONPATH'

ENV KERAS_BACKEND tensorflow

USER keras

# Python
ARG python_version=3.5.1

RUN conda install -y python=${python_version} && \
    conda install scikit-learn six h5py && \
    conda install matplotlib pillow && \
    conda install -y rasterio && \
    conda clean -yt

RUN pip install git+git://github.com/jakebian/quiver.git && \
    pip install flake8 awscli
