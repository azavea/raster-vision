## Preparing datasets

Before running any experiments locally, the data needs to be prepared so that Keras can consume it. This involves splitting large images into tiles and saving them in a specific directory structure. For the
2D Semantic Labeling Contest Vaihingen dataset, which involves aerial imagery, retrieve data from
 http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html Then run `python process_data.py`. This will generate `/opt/data/processed_vaihingen`.
 To upload this data to S3 so it can be used on EC2, upload a zip file of this directory named `processed_vaihingen.zip` to the `otid-data` bucket.

## Running experiments

An experiment is a run of an algorithm on a specific dataset and set of parameters. Each experiment is defined using an options `json` file of the following form
```json
{
    "batch_size": 32,
    "git_commit": "7c8755dcf9a22089c8a53a1a0ca702ae20c208d2",
    "input_shape": [
        256,
        256,
        3
    ],
    "model_type": "fcn_vgg",
    "nb_epoch": 2,
    "nb_labels": 6,
    "nb_prediction_images": 8,
    "nb_val_samples": 256,
    "patience": 3,
    "samples_per_epoch": 256
}
```
In order to run an experiment, you must also provide a list of tasks to perform. These tasks
include `setup` (create unique `run_name` and setup directory structure), `train` (train the model on data) and `eval` (evaluate the learned model and generate visualizations). The run command can be
invoked as follows.
```shell
python run.py <options_file_name> <list of tasks>
# eg. python options.json setup train eval
# eg. python options.json eval
```
This will generate a directory structure in `/opt/data/results/<run_name>/` which contains the options file, the learned model, and various metrics and visualization files.
