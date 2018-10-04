import os

import rastervision as rv


class ObjectDetectionIntegrationTest(rv.ExperimentSet):
    def exp_main(self, tmp_dir):
        def get_path(part):
            return os.path.join(os.path.dirname(__file__), part)

        img_path = get_path('scene/image.tif')
        label_path = get_path('scene/labels.json')
        backend_conf_path = get_path('configs/backend.config')

        pretrained_model = ('https://github.com/azavea/raster-vision-data/'
                            'releases/download/v0.0.5/od-model.tar.gz')

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({
                                'car': (1, 'blue'),
                                'building': (2, 'red')
                            }) \
                            .with_chip_options(neg_ratio=0.0,
                                               ioa_thresh=1.0,
                                               window_method='sliding') \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                  .with_task(task) \
                                  .with_num_steps(350) \
                                  .with_template(backend_conf_path) \
                                  .with_pretrained_model(pretrained_model) \
                                  .with_train_options(sync_interval=None,
                                                      do_monitoring=False,
                                                      replace_model=True) \
                                  .build()

        scene = rv.SceneConfig.builder() \
                              .with_task(task) \
                              .with_id('od_test') \
                              .with_raster_source(img_path, channel_order=[0, 1, 2]) \
                              .with_label_source(label_path) \
                              .build()

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scene(scene) \
                                  .with_validation_scene(scene) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('object-detection-test') \
                                        .with_root_uri(tmp_dir) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_eval_key('default') \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
