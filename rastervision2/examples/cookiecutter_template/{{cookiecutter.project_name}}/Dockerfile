FROM {{cookiecutter.parent_docker_image}}

# The RV image has /opt/src/rastervision2. This repo adds a rastervision2
# directory and to keep the two separate, everything added by this repo is rooted
# at /opt/src/{project_name}.
WORKDIR /opt/src/{{cookiecutter.project_name}}/
COPY rastervision2 /opt/src/{{cookiecutter.project_name}}/rastervision2
COPY rastervision2 /opt/src/{{cookiecutter.project_name}}/rastervision2
COPY scripts /opt/src/{{cookiecutter.project_name}}/scripts
ENV PYTHONPATH=/opt/src/{{cookiecutter.project_name}}:$PYTHONPATH

CMD ["bash"]
