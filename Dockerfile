ARG PYBASE_IMAGE
FROM ${PYBASE_IMAGE}

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
