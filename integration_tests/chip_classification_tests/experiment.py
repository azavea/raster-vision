import os

import rastervision as rv
from integration_tests.util.misc import str_to_bool


class ChipClassificationIntegrationTest(rv.ExperimentSet):
    def exp_main(self, root_uri, data_uri=None, full_train=False, use_tf=False):
        full_train = str_to_bool(full_train)
        use_tf = str_to_bool(use_tf)

        def get_path(part):
            if full_train:
                return os.path.join(data_uri, part)
            else:
                return os.path.join(os.path.dirname(__file__), part)

        img_path = get_path('scene/image.tif')
        label_path = get_path('scene/labels.json')

        img2_path = get_path('scene/image2.tif')
        label2_path = get_path('scene/labels2.json')

        backend_conf_path = get_path('configs/backend.config')

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_chip_size(200) \
                            .with_classes({
                                'car': (1, 'red'),
                                'building': (2, 'blue'),
                                'background': (3, 'black')
                            }) \
                            .with_debug(True) \
                            .build()

        if use_tf:
            pretrained_model = (
                'https://github.com/azavea/raster-vision-data/'
                'releases/download/v0.0.7/chip-classification-test-weights.hdf5')

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
        else:
            if full_train:
                backend = rv.BackendConfig.builder(rv.PYTORCH_CHIP_CLASSIFICATION) \
                    .with_task(task) \
                    .with_train_options(
                        batch_size=16,
                        num_epochs=10,
                        sync_interval=200) \
                    .build()
            else:
                pretrained_uri = (
                    'https://github.com/azavea/raster-vision-data/releases/download/'
                    'v0.9.0/pytorch_chip_classification_test.pth')
                backend = rv.BackendConfig.builder(rv.PYTORCH_CHIP_CLASSIFICATION) \
                    .with_task(task) \
                    .with_train_options(
                        batch_size=8,
                        num_epochs=1) \
                    .with_pretrained_uri(pretrained_uri) \
                    .build()

        def make_scene(i_path, l_path):
            label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                                               .with_uri(l_path) \
                                               .with_ioa_thresh(0.5) \
                                               .with_use_intersection_over_cell(False) \
                                               .with_pick_min_class_id(True) \
                                               .with_background_class_id(3) \
                                               .with_infer_cells(True) \
                                               .build()

            raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                                 .with_uri(i_path) \
                                                 .with_channel_order([0, 1, 2]) \
                                                 .with_stats_transformer() \
                                                 .build()

            return rv.SceneConfig.builder() \
                                 .with_task(task) \
                                 .with_id(os.path.basename(i_path)) \
                                 .with_raster_source(raster_source) \
                                 .with_label_source(label_source) \
                                 .build()

        scene_1 = make_scene(img_path, label_path)
        scene_2 = make_scene(img2_path, label2_path)

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes([scene_1, scene_2]) \
                                  .with_validation_scenes([scene_1, scene_2]) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('chip-classification-test') \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_stats_analyzer() \
                                        .with_eval_key('default') \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
