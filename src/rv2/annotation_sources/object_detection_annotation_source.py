from rv2.core.annotation_source import AnnotationSource
from rv2.annotations.object_detection_annotations import (
    ObjectDetectionAnnotations)


class ObjectDetectionAnnotationSource(AnnotationSource):
    def get_annotations(self, window, ioa_thresh=1.0):
        return self.annotations.get_subset(
            window, ioa_thresh=ioa_thresh)

    def get_all_annotations(self):
        return self.annotations

    def extend(self, window, annotations):
        self.annotations = self.annotations.concatenate(
            window, annotations)

    def post_process(self, options):
        self.annotations = self.annotations.prune_duplicates(
            options.object_detection_options.score_thresh,
            options.object_detection_options.merge_thresh)

    def clear(self):
        self.annotations = ObjectDetectionAnnotations.make_empty()
