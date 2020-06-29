FROM {{cookiecutter.parent_docker_image}}

# Uncomment if you add any non-RV requirements.
# COPY ./rastervision_{{cookiecutter.project_name}}/requirements.txt /opt/src/requirements.txt
# RUN pip install $(grep -ivE "rastervision_*" requirements.txt)

COPY ./rastervision_{{cookiecutter.project_name}}/ /opt/src/rastervision_{{cookiecutter.project_name}}/

ENV PYTHONPATH=/opt/src/rastervision_{{cookiecutter.project_name}}/:$PYTHONPATH

CMD ["bash"]
