from typing import List, Optional

from rastervision2.pipeline.config import (Config, register_config,
                                           ConfigError, Field)


@register_config('class_config')
class ClassConfig(Config):
    """Configures the class names that are being predicted."""
    names: List[str] = Field(..., description='Names of classes.')
    colors: List[str] = Field(
        ..., description='Colors used to visualize classes.')
    null_class: Optional[str] = Field(
        None,
        description=
        ('Optional name of class in `names` to use as the null class. This is used in '
         'semantic segmentation to represent the label for imagery pixels that are '
         'NODATA or that are missing a label.'))

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
        if self.null_class is None:
            self.null_class = 'null'
            self.names.append('null')
            self.colors.append('black')

    def update(self, pipeline=None):
        pass

    def validate_config(self):
        if self.null_class is not None and self.null_class not in self.names:
            raise ConfigError(
                'The null_class: {} must be in list of class names.'.format(
                    self.null_class))

    def __len__(self):
        return len(self.names)
