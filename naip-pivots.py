import ...


def get_scenes_to_labels():
    """ Construct a list of tuples of S3 URI's to scenes, and the S3 URI to the label GeoJSON
    """
    pass

def split_scenes(scenes_to_labels, test_ratio=0.8):
    """ Read in the scenes_to_labels, read GeoJSON labels and split according to the
        ratio. Returns a tuple  of lists (train,  val)
    """
    pass

class NaipExperiments(rastervision.Experiments):

    ## Three big concepts: Datasets, Tasks, and Backends
    ## Datasets consist of Scenes (LabelSource, LabelStore, RasterSource (Transform), Augmentation)
    ## Option configuration for each command:
    ##  Analyze
    ##  Chip
    ##  Train
    ##  Predict
    ##  Eval

    def  dataset(self, task):
        (train_scenes, val_scenes) = split_scenes(get_scenes_to_labels())

        def make_labeled_scene_config(uris, predict=False):
            result = []
            for (scene_uri, label_uri) in uris:
                raster_source = RasterSourceConfig.builder(rv.GEOTIFF) \
                                                  .with_uri(uri) \
                                                  .with_channel_order([0,1,2]) \
                                                  .with_stats_transform() \
                                                  .build()

                label_source = LabelSourceConfig.builder(task, rv.OBJECT_DETECTION_GEOJSON) \
                                                .with_task(task)
                                                .with_uri(uri) \
                                                .build()

                s = SceneConfig.builder() \
                               .with_raster_source(raster_source) \
                               .with_label_source(label_source)

                if predict:
                    label_store = LabelStoreConfig.builder(task, rv.OBJECT_DETECTION_GEOJSON) \
                                                  .with_uri(uri) \
                                                  .build()
                    s.with_label_store(label_store)
                s.build()

        augmentation =  AugmentationConfig.builder(task, rv.IMAGEAUG) \
                                          .with_config({
                                              "scale": True
                                          }) \
                                          .build()

        return DatasetConfig.builder(task) \
                            .with_train_scenes(map(make_labeled_scene_config, train_scenes)) \
                            .with_val_scenes(map(lambda x: make_labeled_scene_config(x, True),
                                                 val_scenes)) \
                            .with_augmentation(augmentation) \
                            .build()

    def base_experiment(self):

        task = TaskConfig.builder(OD) \
                         .with_classes(["pivot", "something"], background_class="something") \
                         .with_chip_size(300) \
                         .build()

        backend = BackendConfig.builder(TFOD) \
                               .for_task(t) \
                               .with_model_config({
                                   "score_converter": "SOFTMAX"
                               }) \
                               .with_files({
                                   "train.py" : "/opt/tf-models/object_detection/train.py"
                               }) \
                               .with_train_config({
                                   "num_steps": num_steps,
                                   "batch_size": 8
                               }) \
                               .with_pretrained_model("*.zip") \
                               .with_sync_interval(500) \
                               .build()

        # Command Options
        predict_opts = PredictOptions.builder(task) \
                                .with_merge_thresh(0.2) \
                                .build()

        eval_conf = EvalOptions()...

        chip = ChipOptions()...

        #  Can we make 1 word?
        compute_stats = ComputeStatsConfig()...

        e = ExperimentConfig.builder() \
                            .with_task(tasks) \
                            .with_dataset(self.dataset(task)) \
                            .with_backend(backend) \
                            .with_predict_options(predict_opts)
        # Eval defaults

        return e.with_name("NAIP center pivot") \
                .with_root("s3://raster-vision-rob-dev/naip/center-pivot/od")

    def experiment_naip(num_steps=10):
        e = ExperimentConfig.load_from("exp.json")
        p = ExperimentConfig.predict.builder()\
                                    .with_predict_scenes(new_scenes) \
                                    .build()

        e = e.builder() \
             .with_predict(p)
             .with_uris(predict="s3://raster-vision-rob-dev/naip/center-pivot/od/chips") \
             .build()

        return e

    def experiment_prediction():
        e = base_experiment()
        p = ExperimentConfig.predict.builder() \
                                    .with_predict_scenes(new_scenes) \
                                    .build()

        return e.builder() \
                .with_predict(p) \
                .build()


if __name__ == "__main__":
    rastervision.main()

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
