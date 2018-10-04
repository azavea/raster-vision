import os

import rastervision as rv


class ChipClassificationIntegrationTest(rv.ExperimentSet):
    def exp_main(self, tmp_dir):
        def get_path(part):
            return os.path.join(os.path.dirname(__file__), part)

        img_path = get_path('scene/image.tif')
        label_path = get_path('scene/labels.json')
        backend_conf_path = get_path('configs/backend.config')

        pretrained_model = (
            'https://github.com/fchollet/'
            'deep-learning-models/releases/download/v0.2/'
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_chip_size(200) \
                            .with_classes({
                                'car': (1, 'red'),
                                'building': (2, 'blue'),
                                'background': (3, 'black')
                            }) \
                            .with_debug(True) \
                            .build()

        backend = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                  .with_task(task) \
                                  .with_debug(True) \
                                  .with_template(backend_conf_path) \
                                  .with_num_epochs(8) \
                                  .with_pretrained_model(pretrained_model) \
                                  .with_train_options(sync_interval=None,
                                                      do_monitoring=False,
                                                      replace_model=True) \
                                  .build()

        label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                                           .with_uri(label_path) \
                                           .with_ioa_thresh(0.5) \
                                           .with_use_intersection_over_cell(False) \
                                           .with_pick_min_class_id(True) \
                                           .with_background_class_id(3) \
                                           .with_infer_cells(True) \
                                           .build()

        raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                             .with_uri(img_path) \
                                             .with_channel_order([0, 1, 2]) \
                                             .with_stats_transformer() \
                                             .build()

        scene = rv.SceneConfig.builder() \
                              .with_task(task) \
                              .with_id('cc_test') \
                              .with_raster_source(raster_source) \
                              .with_label_source(label_source) \
                              .build()

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scene(scene) \
                                  .with_validation_scene(scene) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('chip-classification-test') \
                                        .with_root_uri(tmp_dir) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_stats_analyzer() \
                                        .with_eval_key('default') \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
