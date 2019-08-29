import os

import rastervision as rv


class SemanticSegmentationIntegrationTest(rv.ExperimentSet):
    def exp_main(self, root_uri, use_tf=False):
        def get_path(part):
            return os.path.join(os.path.dirname(__file__), part)

        img_paths = [get_path('scene/image.tif'), get_path('scene/image2.tif')]
        label_paths = [
            get_path('scene/labels.tif'),
            get_path('scene/labels2.tif')
        ]
        class_map = {'red': (1, 'red'), 'green': (2, 'green')}
        num_steps = 1
        batch_size = 1

        # This a divisor of the scene length.
        chip_size = 300
        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(chip_size) \
                            .with_classes(class_map) \
                            .with_chip_options(window_method='sliding',
                                               stride=chip_size,
                                               debug_chip_probability=1.0) \
                            .build()

        if use_tf:
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

            # .with_config below needed to copy final layer from pretrained model.
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
        else:
            # TODO
            pretrained_uri = ''
            num_epochs = 1
            backend = rv.BackendConfig.builder(rv.FASTAI_SEMANTIC_SEGMENTATION) \
                .with_task(task) \
                .with_train_options(
                    lr=1e-4,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    model_arch='resnet18',
                    debug=False) \
                .with_pretrained_uri(pretrained_uri) \
                .build()

        label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                           .with_rgb_class_map(task.class_map) \
                                           .with_raster_source(label_paths[0]) \
                                           .build()

        label_source2 = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                            .with_rgb_class_map(task.class_map) \
                                            .with_raster_source(label_paths[1]) \
                                            .build()

        vector_output = [{
            'mode': 'buildings',
            'class_id': 1,
            'building_options': {
                'element_width_factor': 0.51
            }
        }, {
            'denoise': 50,
            'mode': 'polygons',
            'class_id': 1
        }]

        label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                         .with_vector_output(vector_output) \
                                         .with_rgb(True) \
                                         .build()

        label_store2 = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                         .with_vector_output(vector_output) \
                                         .with_rgb(True) \
                                         .build()

        scene = rv.SceneConfig.builder() \
                              .with_task(task) \
                              .with_id('test-scene') \
                              .with_raster_source(img_paths[0], channel_order=[0, 1, 2]) \
                              .with_label_source(label_source) \
                              .with_label_store(label_store) \
                              .build()

        scene2 = rv.SceneConfig.builder() \
                              .with_task(task) \
                              .with_id('test-scene2') \
                              .with_raster_source(img_paths[1], channel_order=[0, 1, 2]) \
                              .with_label_source(label_source2) \
                              .with_label_store(label_store2) \
                              .build()

        scenes = [scene, scene2]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(scenes) \
                                  .with_validation_scenes(scenes) \
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
