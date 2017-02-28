## Data directory structure

All data including datasets and results are stored in a single directory outside of the repo. The `Vagrantfile` maps `~/data` on the host machine to `/opt/data` on the guest machine. The datasets are stored in `/opt/data/datasets` and results are stored in `/opt/data/results`.

## Preparing datasets

Before running any experiments locally, the data needs to be prepared so that Keras can consume it. This involves splitting large images into tiles and saving them in a specific directory structure. For the
[ISPRS 2D Semantic Labeling Potsdam dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html), you can download the data after filling out the [request form](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html).
After following the link to the Potsdam dataset, download
`1_DSM_normalisation.zip`, `3_Ortho_IRRG.zip`, and `5_Labels_for_participants.zip`. Then unzip the files into
`/opt/data/datasets/potsdam`, resulting in `/opt/data/datasets/potsdam/1_DSM_normalisation/`, etc.

Then run `python -m model_training.data.preprocess`. This will generate `/opt/data/datasets/processed_potsdam`. As a test, you may want to run `python -m model_training.data.generators` which will generate a PDF visualizing a minibatch in `/opt/data/datasets/processed_potsdam/train/batch_0.pdf`.
 To make the processed data available for use on EC2, upload a zip file of `/opt/data/datasets/processed_potsdam` named `processed_potsdam.zip` to the `otid-data` bucket.

## Running experiments

An experiment consists of training a model on a dataset using a set of hyperparameters. Each experiment is defined using an options `json` file of the following form
```json
{
    "batch_size": 1,
    "patience": 10,
    "dataset": "potsdam",
    "git_commit": "d5ae66",
    "include_depth": false,
    "kernel_size": [5, 5],
    "is_big_model": false,
    "model_type": "conv_logistic",
    "nb_epoch": 2,
    "nb_labels": 6,
    "nb_val_samples": 1,
    "run_name": "2_28_17/conv_logistic_test",
    "samples_per_epoch": 2,
    "nb_eval_samples": 1
}
```
It is a good idea to include the `git_commit` field and commit the options files to `git` so that the experiment is repeatable. In order to run an experiment, you must also provide a list of tasks to perform. These tasks
include `setup` (setup directory structure for results), `train` (train the model on data) and `eval` (evaluate the learned model and generate visualizations). Here are some examples of how to use the `run` command.
```shell
# Run all tasks
python -m model_training.run experiments/2_28_17/conv_logistic_test.json setup train eval
# Run all tasks by default
python -m model_training.run experiments/2_28_17/conv_logistic_test.json
# Only run the eval task, which assumes setup and train were run previously
python -m model_training.run experiments/2_28_17/conv_logistic_test.json eval
```
This will generate a directory structure in `/opt/data/results/<run_name>/` which contains the options file, the learned model, and various metrics and visualization files. I have been using `<date>/<short experiment description>`
as a convention for the `run_name` field.
