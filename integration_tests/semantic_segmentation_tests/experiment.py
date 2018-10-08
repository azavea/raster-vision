import os

import rastervision as rv


class SemanticSegmentationIntegrationTest(rv.ExperimentSet):
    def exp_main(self, root_uri):
        def get_path(part):
            return os.path.join(os.path.dirname(__file__), part)

        img_path = get_path('scene/image.tif')
        label_path = get_path('scene/labels.tif')
        class_map = {'red': (1, 'red'), 'green': (2, 'green')}
        num_steps = 1
        batch_size = 1

        # These are the parameters that were used to train the following pretrained
        # model.
        # num_steps = 5000
        # batch_size = 8
        # I found it was also possible to train for 2000 steps with the same eval,
        # and even fewer steps may be possible. But because it takes
        # 5 secs/step with batch size of 1 on a CPU, it doesn't seem feasible to actually
        # train during CI. So instead we just use the model trained on the GPU and then
        # fine-tune it for one step.
        pretrained_model = (
            'https://github.com/azavea/raster-vision-data/releases/'
            'download/0.0.6/deeplab-test-model.tar.gz')

        # This a divisor of the scene length.
        chip_size = 300
        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(chip_size) \
                            .with_classes(class_map) \
                            .with_chip_options(window_method='sliding',
                                               stride=chip_size,
                                               debug_chip_probability=1.0) \
                            .build()

        # .with_config belowe needed to copy final layer from pretrained model.
        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(rv.MOBILENET_V2) \
                                  .with_pretrained_model(pretrained_model) \
                                  .with_train_options(do_monitoring=True,
                                                      replace_model=True) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(True) \
                                  .with_config({'initializeLastLayer': 'true'}) \
                                  .build()

        label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                           .with_rgb_class_map(task.class_map) \
                                           .with_raster_source(label_path) \
                                           .build()

        label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                         .with_rgb(True) \
                                         .build()

        scene = rv.SceneConfig.builder() \
                              .with_task(task) \
                              .with_id('test-scene') \
                              .with_raster_source(img_path, channel_order=[0, 1, 2]) \
                              .with_label_source(label_source) \
                              .with_label_store(label_store) \
                              .build()

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scene(scene) \
                                  .with_validation_scene(scene) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('semantic-segmentation-test') \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_eval_key('default') \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
