from typing import List, Optional, Union

from rastervision.pipeline.config import (Config, register_config, ConfigError,
                                          Field)
from rastervision.core.data.utils import color_to_triple


@register_config('class_config')
class ClassConfig(Config):
    """Configures the class names that are being predicted."""
    names: List[str] = Field(..., description='Names of classes.')
    colors: Optional[List[Union[List, str]]] = Field(
        None,
        description=
        ('Colors used to visualize classes. Can be color strings accepted by '
         'matplotlib or RGB tuples. If None, a random color will be auto-generated '
         'for each class.'))
    null_class: Optional[str] = Field(
        None,
        description=
        ('Optional name of class in `names` to use as the null class. This is used in '
         'semantic segmentation to represent the label for imagery pixels that are '
         'NODATA or that are missing a label. If None, and this Config is part of a '
         'SemanticSegmentationConfig, a null class will be added automatically.'
         ))

    def get_class_id(self, name):
        return self.names.index(name)

    def get_name(self, id):
        return self.names[id]

    def get_null_class_id(self):
        if self.null_class is None:
            raise ValueError('null_class is not set')
        return self.get_class_id(self.null_class)

    def get_color_to_class_id(self):
        return dict([(self.colors[i], i) for i in range(len(self.colors))])

    def ensure_null_class(self):
        """Add a null class if one isn't set."""
        if self.null_class is None:
            self.null_class = 'null'
            self.names.append('null')
            self.colors.append('black')

    def update(self, pipeline=None):
        if not self.colors:
            self.colors = [color_to_triple() for _ in self.names]

    def validate_config(self):
        if self.null_class is not None and self.null_class not in self.names:
            raise ConfigError(
                'The null_class: {} must be in list of class names.'.format(
                    self.null_class))

    def __len__(self):
        return len(self.names)
