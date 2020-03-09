from typing import List, Optional

from rastervision2.pipeline.config import (Config, register_config,
                                           ConfigError)


@register_config('class_config')
class ClassConfig(Config):
    names: List[str]
    colors: List[str]
    null_class: Optional[str] = None

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
