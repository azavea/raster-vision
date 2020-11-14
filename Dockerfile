ARG BASE_IMAGE
FROM ${BASE_IMAGE}

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
