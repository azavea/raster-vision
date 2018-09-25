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

class CustomTaskConfig(rv.TaskConfig):
    def __init__(self):
        super().__init__("my_task")
        self.classes = {}
        self.chip_size = 300

    def builder(self):
        return CustomTaskConfigBuilder(self)

    def to_proto(self):
        conf = TaskConfigMsg.CustomConfig({})
        return TaskConfigMsg(key="my_task",
                             custom_config=conf)

class CustomTaskConfigBuilder:
    def __init__(self, task_config=None):
        super().__init__(task_config or CustomTaskConfig())

    def from_proto(self, msg):
        conf = msg.custom_config
        b = CustomTaskConfigBuilder()
        return b.with_conf(conf)

    def with_conf(self, conf):
        t = deepcopy(self.task)
        t.conf = conf
        return CustomTaskConfigBuilder(t)

class CustomTask(Task):
    def get_train_windows(self, scene, options):
        pass

    def get_train_labels(self, window, scene, options):
        pass

    def get_predict_windows(self, extent, options):
        pass

    def post_process_predictions(self, labels, options):
        pass

    def get_evaluation(self):
        pass

    def save_debug_predict_image(self, scene, debug_dir_uri):
        pass


class NaipExperiments(rv.Experiments):

    def register_plugins(self):
        rv.plugins.register_task("custom_task", CustomTask, CustomTaskConfigBuilder)
        rv.plugins.register_backend("custom_backend",  CustomBackend)
        rv.plugins.register_data_augmenter("custom_data_augmenter", CustomDataAugmenter)
        rv.plugins.register_eval_metrics("custom_eval_metrics", CustomEvalMetrics)

    def exp_naip_center_pivot_od_mobilenet(self, num_steps=20):

        (train_scenes, val_scenes) = split_scenes(get_scenes_to_labels())

        dataset = rv.DatasetConfig.builder(task) \
                                  .with_train_scenes(SceneConfig.build_from(train)) \
                                  .with_val_scenes(SceneConfig.build_from(val)) \
                                  .build()

        task = rv.TaskConfig.builder(OD) \
                            .with_classes(["pivot"]) \
                            .with_chip_size(300) \
                            .build()

        backend = rv.BackendConfig.builder(TFOD) \
                                  .for_task(t) \
                                  .with_model(MOBILENET) \
                                  .with_train_config({
                                      "num_steps": num_steps,
                                      "batch_size": 8
                                  }) \
                                  .with_pretrained_model("coco") \
                                  .build()


        e = rv.ExperimentConfig.builder() \
                               .with_task(tasks) \
                               .with_dataset(self.dataset()) \
                               .with_backend(backend)

        return e.with_name("NAIP center pivot - mobilenet") \
                .with_root("s3://raster-vision-rob-dev/naip/center-pivot/od") \
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
