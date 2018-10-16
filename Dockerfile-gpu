FROM raster-vision-base:latest

COPY scripts/rastervision /usr/local/bin/
COPY rastervision /opt/src/rastervision
COPY scripts/compile /opt/src/scripts/compile
RUN /opt/src/scripts/compile

CMD ["bash"]
