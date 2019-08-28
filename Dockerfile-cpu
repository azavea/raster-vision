FROM raster-vision-base:latest

RUN pip uninstall -y tensorflow_gpu && pip install tensorflow==1.10.1

COPY scripts/rastervision /usr/local/bin/
COPY rastervision /opt/src/rastervision
COPY scripts/compile /opt/src/scripts/compile
RUN /opt/src/scripts/compile

CMD ["bash"]
