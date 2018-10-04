import os

import rastervision as rv


class SemanticSegmentationIntegrationTest(rv.ExperimentSet):
    def exp_main(self, root_uri):
        def get_path(part):
            return os.path.join(os.path.dirname(__file__), part)

        # img_path = get_path('scene/image.tif')
        # label_path = get_path('scene/labels.tif')

        img_path = ('s3://raster-vision-rob-dev/integration-tests/'
                    'semantic_segmentation_tests/scene/image.tif')
        label_path = ('s3://raster-vision-rob-dev/integration-tests/'
                      'semantic_segmentation_tests/scene/labels.tif')

        class_map = {'car': (1, 'red'), 'building': (2, 'green')}

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(256) \
                            .with_classes(class_map) \
                            .with_chip_options(window_method='sliding',
                                               stride=256) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(rv.MOBILENET_V2) \
                                  .with_train_options(do_monitoring=True,
                                                      replace_model=True) \
                                  .with_num_steps(30000) \
                                  .with_batch_size(8) \
                                  .with_debug(True) \
                                  .with_config({'saveIntervalSecs': 5}) \
                                  .with_config({'saveSummariesSecs': 5}) \
                                  .build()

        label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                           .with_source_class_map(task.class_map) \
                                           .with_raster_source(label_path) \
                                           .build()

        label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
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
