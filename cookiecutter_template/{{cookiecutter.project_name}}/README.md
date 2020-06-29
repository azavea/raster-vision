# {{cookiecutter.project_name}}

{{cookiecutter.description}}

## Running locally

### Setup

* Build the Docker image using `docker/build`.
* Set any environment variables needed by `docker/run`, which you can read about by running `docker/run --help`.

### Usage

* Get a console in the container using `docker/run --aws`
* The code in `rastervision/{{cookiecutter.project_name}}` contains a very simple plugin with a `TestPipeline`. You can run the test pipeline included in the source code using
 `rastervision run inprocess rastervision.{{cookiecutter.project_name}}.configs.test`

## Running on AWS Batch

### Setup

* If using Batch, create a new job definition stack using the [CloudFormation template](https://github.com/azavea/raster-vision-aws#deploy-new-job-definitions). This is needed because there should be a CPU and GPU job def for each project / user pair. These job definitions will point to an ECR repo:tag used to store your Docker image used when running remotely.
* Create a Raster Vision profile file in `~/.rastervision/{{cookiecutter.project_name}}` with the job definitions that were created along with the job queues which already exist.
* Set the {{cookiecutter.caps_project_name}}_ECR_IMAGE environment variable to the ECR repo with the namespace tag you used in the previous step, ie. `repo:useridProjectName`. This env var needs to be set to run the `docker/ecr_publish` script.

### Usage

* Run `docker/build` and then `docker/ecr_publish` to push any changes to ECR.
* To run the pipeline on Batch, use the `batch` runner, the `--profile` option with the Raster Vision profile you created above, and set the `root_uri` to an S3 URI. This can be done using
 `rastervision --profile {{cookiecutter.project_name}} rastervision.{{cookiecutter.project_name}}.configs.test run batch  -a root_uri <s3 uri>`
