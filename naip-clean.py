import rastervision as rv

def get_scenes_to_labels():
    """ Construct a list of tuples of S3 URI's to scenes, and the S3 URI to the label GeoJSON
    """
    pass

def split_scenes(scenes_to_labels, test_ratio=0.8):
    """ Read in the scenes_to_labels, read GeoJSON labels and split according to the
        ratio. Returns a tuple  of lists (train,  val)
    """
    pass

class NaipExperiments(rv.Experiments):
    def exp_naip_center_pivot_mobilenet(self, num_steps=20):
        (train_scenes, val_scenes) = split_scenes(get_scenes_to_labels())

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                         .with_classes(["pivot"]) \
                         .with_chip_size(300) \
                         .build()

        dataset = rv.DatasetConfig.builder() \
                                  .with_task(task) \
                                  .with_train_scenes(train_scenes) \
                                  .with_val_scenes(val_scenes) \
                                  .build()


        backend = rv.BackendConfig.builder(rv.TFOD) \
                               .with_task(task) \
                               .with_model(rv.SSD_MOBILENET_V1_COCO) \
                               .build()


        e = rv.ExperimentConfig.builder() \
                            .with_task(tasks) \
                            .with_dataset(dataset) \
                            .with_backend(backend)

        return e.with_name("NAIP center pivot - mobilenet") \
                .with_root("s3://raster-vision-rob-dev/naip/center-pivot/") \
                .build()

if __name__ == "__main__":
    rv.main()

## To run, similar to running unit tests.
#
# --profile specifies the raster vision profile, containing configuration such as AWS Batch settings,
# works much like aws-cli
#
# e.g.
#
# python naip-pivots.py *mobilenet run --remote --profile naip
# # Or from root of some repo containing experiments
# rv run *naip*mobilenet--remote --profile naip



python naip-pivots.py *naip*2 run --remote --profile naip


rastervision  exp.json



## To run:

## By default, raster vision will look for and call
## any method in the file that starts with "experiment".
## This can return a single or multiple experiments.

# Using default profile
# > rastervision run naip-pivots.py --remote
#
# Using custom profile
# > rastervision run naip-pivots.py --remote --profile naip
#
# Executing specific  commands
# > rastervision run naip-pivots.py --remote train predict eval
#
# Running a specific method, and passing arguments to the method
# > rastervision run naip-pviots.py --method experiment_naip --args num_steps=40 --remote train predict eval
w
