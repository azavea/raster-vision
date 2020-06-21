FROM {{cookiecutter.parent_docker_image}}

# The RV image has /opt/src/rastervision. This repo adds a rastervision
# directory and to keep the two separate, everything added by this repo is rooted
# at /opt/src/{project_name}.
WORKDIR /opt/src/{{cookiecutter.project_name}}/
COPY rastervision /opt/src/{{cookiecutter.project_name}}/rastervision
COPY configs /opt/src/{{cookiecutter.project_name}}/configs
COPY scripts /opt/src/{{cookiecutter.project_name}}/scripts
ENV PYTHONPATH=/opt/src/{{cookiecutter.project_name}}:$PYTHONPATH

CMD ["bash"]
