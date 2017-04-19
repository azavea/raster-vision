# Based on https://github.com/fchollet/keras/blob/2b51317be82d4420169d2cc79dc4443028417911/docker/Dockerfile
FROM keras-semantic-segmentation-base

USER keras

# Python

ARG tensorflow_version=0.10.0-cp35-cp35m
ARG architecture=gpu

RUN pip install https://storage.googleapis.com/tensorflow/linux/${architecture}/tensorflow-${tensorflow_version}-linux_x86_64.whl && \
    pip install git+git://github.com/fchollet/keras.git@4fa7e5d454dd4f3f33f1d756a2a8659f2e789141

WORKDIR /opt/src

COPY . /opt/src

# Ensure that the keras user will have permission to write model into /opt/data
USER root
RUN mkdir /opt/data
RUN chown -R keras:root /opt/data

CMD ["bash"]
