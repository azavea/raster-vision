from typing import Dict, Optional

from rastervision.v2.core.config import Config

class VectorSourceConfig(Config):
    default_class_id: int = 0
    class_id_to_filter: Optional[Dict] = None
    line_bufs: Optional[Dict] = None
    point_bufs: Optional[Dict] = None

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
