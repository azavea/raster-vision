from typing import Dict, Optional, Union

from rastervision2.pipeline.config import Config, register_config


@register_config('vector_source')
class VectorSourceConfig(Config):
    default_class_id: int = 0
    class_id_to_filter: Optional[Dict[int, Optional[Dict]]] = None
    line_bufs: Optional[Dict[int, Union[int, float, None]]] = None
    point_bufs: Optional[Dict[int, Union[int, float, None]]] = None

    def has_null_class_bufs(self):
        if self.point_bufs is not None:
            for c, v in self.point_bufs.items():
                if v is None:
                    return True

        if self.line_bufs is not None:
            for c, v in self.line_bufs.items():
                if v is None:
                    return True

        return False

    def build(self, class_config, crs_transformer):
        raise NotImplementedError()

    def update(self, pipeline=None, scene=None):
        pass
