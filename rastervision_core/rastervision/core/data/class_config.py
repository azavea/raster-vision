from typing import List, Optional, Tuple, Union

from rastervision.pipeline.config import (Config, register_config, ConfigError,
                                          Field, validator)
from rastervision.core.data.utils import color_to_triple, normalize_color

DEFAULT_NULL_CLASS_NAME = 'null'
DEFAULT_NULL_CLASS_COLOR = 'black'


@register_config('class_config')
class ClassConfig(Config):
    """Configure class information for a machine learning task."""

    names: List[str] = Field(
        ...,
        description='Names of classes. The i-th class in this list will have '
        'class ID = i.')
    colors: Optional[List[Union[str, Tuple]]] = Field(
        None,
        description=
        ('Colors used to visualize classes. Can be color strings accepted by '
         'matplotlib or RGB tuples. If None, a random color will be auto-generated '
         'for each class.'))
    null_class: Optional[str] = Field(
        None,
        description='Optional name of class in `names` to use as the null '
        'class. This is used in semantic segmentation to represent the label '
        'for imagery pixels that are NODATA or that are missing a label. '
        f'If None and the class names include "{DEFAULT_NULL_CLASS_NAME}", '
        'it will automatically be used as the null class. If None, and this '
        'Config is part of a SemanticSegmentationConfig, a null class will be '
        'added automatically.')

    @validator('colors', always=True)
    def validate_colors(cls, v: Optional[List[Union[str, Tuple]]],
                        values: dict) -> Optional[List[Union[str, Tuple]]]:
        """Compare length w/ names. Also auto-generate if not specified."""
        class_names = values['names']
        class_colors = v
        if class_colors is None:
            class_colors = [color_to_triple() for _ in class_names]
        elif len(class_names) != len(class_colors):
            raise ConfigError(f'len(class_names) ({len(class_names)}) != '
                              f'len(class_colors) ({len(class_colors)})\n'
                              f'class_names: {class_names}\n'
                              f'class_colors: {class_colors}')
        return class_colors

    @validator('null_class', always=True)
    def validate_null_class(cls, v: Optional[str],
                            values: dict) -> Optional[str]:
        """Check if in names. If 'null' in names, use it as null class."""
        names = values['names']
        if v is None:
            if DEFAULT_NULL_CLASS_NAME in names:
                v = DEFAULT_NULL_CLASS_NAME
        else:
            if v not in names:
                raise ConfigError(
                    f'The null_class, "{v}", must be in list of class names.')

            # edge case
            default_null_class_in_names = (DEFAULT_NULL_CLASS_NAME in names)
            null_class_neq_default = (v != DEFAULT_NULL_CLASS_NAME)
            if default_null_class_in_names and null_class_neq_default:
                raise ConfigError(
                    f'"{DEFAULT_NULL_CLASS_NAME}" is in names but the '
                    f'specified null_class is something else ("{v}").')
        return v

    def get_class_id(self, name: str) -> int:
        return self.names.index(name)

    def get_name(self, id: int) -> str:
        return self.names[id]

    @property
    def null_class_id(self) -> int:
        if self.null_class is None:
            raise ValueError('null_class is not set')
        return self.get_class_id(self.null_class)

    def get_color_to_class_id(self) -> dict:
        return dict([(self.colors[i], i) for i in range(len(self.colors))])

    def ensure_null_class(self) -> None:
        """Add a null class if one isn't set. This method is idempotent."""
        if self.null_class is not None:
            return

        null_class_name = DEFAULT_NULL_CLASS_NAME
        null_class_color = DEFAULT_NULL_CLASS_COLOR

        # This might seeem redundant given the null class validator above, but
        # is actually important. Sometimes there can be multiple ClassConfig
        # instances that reference the same list objects for names and colors
        # (not clear why this happens). This means that
        # each ensure_null_class() call will add to names and colors in each
        # copy of ClassConfig but only set its own null_class, which makes this
        # method() non-idempotent.
        if null_class_name in self.names:
            self.null_class = null_class_name
            return

        # use random color if default color is already taken
        null_class_color_triple = color_to_triple(null_class_color)
        all_color_triples = [
            color_to_triple(c) if isinstance(c, str) else c
            for c in self.colors
        ]
        if null_class_color_triple in all_color_triples:
            null_class_color = color_to_triple()

        self.names.append(null_class_name)
        self.colors.append(null_class_color)
        self.null_class = null_class_name

    def __len__(self) -> int:
        return len(self.names)

    @property
    def color_triples(self) -> List[Tuple[float, float, float]]:
        """Class colors in a normalized form."""
        color_triples = [normalize_color(c) for c in self.colors]
        return color_triples
