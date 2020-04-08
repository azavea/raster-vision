# {{cookiecutter.docker_image}}

## Usage

* Build the Docker image using `./docker/build`.
* Set any environment variables needed by `./docker/run`, which you can read about by running `./docker/run --help`.
* Get a console in the container using `./docker/run --aws`
* The code in `rastervision2/{{cookiecutter.project_name}}` contains a very simple plugin with a `TestPipeline`. You can run the test pipeline included in the source code using:
```
root@0208ac5dee52:/opt/src/{{cookiecutter.project_name}}# rastervision2 run inprocess configs/test.py
Running print_msg command...
hello world
```
