from rv2.core.ml_task import MLTask


class Classification(MLTask):
    def get_train_windows(self, project, options):
        imagery_extent = project.raster_source.get_extent()
        chip_size = options.chip_size
        stride = chip_size
        return imagery_extent.get_windows(chip_size, stride)

    def get_train_labels(self, window, project, options):
        return project.ground_truth_label_source.get_labels(window)

    def get_predict_windows(self, extent, options):
        pass

    def get_evaluation(self):
        pass
