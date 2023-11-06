from typing import List, Optional
import click

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.predictor import Predictor, ScenePredictor


# https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click
class OptionEatAll(click.Option):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def is_next_option(arg: str) -> bool:
            for prefix in self._eat_all_parser.prefixes:
                if arg.startswith(prefix):
                    return True
            return False

        def parser_process(value, state):
            # method to hook to the parser.process
            values = [value]
            # grab everything up to the next option
            while state.rargs and not is_next_option(state.rargs[0]):
                values.append(state.rargs.pop(0))
            # call the actual process
            self._previous_parser_process(values, state)

        retval = super().add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = (parser._long_opt.get(name)
                          or parser._short_opt.get(name))
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break

        return retval


@click.command(
    'predict', short_help='Use a model bundle to predict on new images.')
@click.argument('model_bundle')
@click.argument('image_uri')
@click.argument('label_uri')
@click.option(
    '--update-stats',
    '-a',
    is_flag=True,
    help=('Run an analysis on this individual image, as '
          'opposed to using any analysis like statistics '
          'that exist in the prediction package'))
@click.option(
    '--channel-order',
    cls=OptionEatAll,
    # https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click#comment121399899_48394004
    type=list,
    help='List of indices comprising channel_order. Example: 2 1 0')
@click.option(
    '--scene-group',
    help='Name of the scene group whose stats will be used by the '
    'StatsTransformer. Requires the stats for this scene group to be present '
    'inside the bundle.')
def predict(model_bundle: str,
            image_uri: str,
            label_uri: str,
            update_stats: bool = False,
            channel_order: Optional[List[str]] = None,
            scene_group: Optional[str] = None):
    """Make predictions on the images at IMAGE_URI
    using MODEL_BUNDLE and store the prediction output at LABEL_URI.
    """
    if channel_order is not None:
        channel_order: List[int] = [int(i) for i in channel_order]

    with get_tmp_dir() as tmp_dir:
        predictor = Predictor(model_bundle, tmp_dir, update_stats,
                              channel_order, scene_group)
        predictor.predict([image_uri], label_uri)


@click.command(
    'predict_scene',
    short_help='Use a model bundle to predict on a new scene.')
@click.argument('model_bundle_uri')
@click.argument('scene_config_uri')
@click.option(
    '--predict_options_uri',
    type=str,
    default=None,
    help='Optional URI to serialized Raster Vision PredictOptions config.')
def predict_scene(model_bundle_uri: str,
                  scene_config_uri: str,
                  predict_options_uri: Optional[str] = None):
    """Use a model-bundle to make predictions on a scene.

    \b
    MODEL_BUNDLE_URI    URI to a serialized Raster Vision model-bundle.
    SCENE_CONFIG_URI    URI to a serialized Raster Vision SceneConfig.
    """
    predictor = ScenePredictor(model_bundle_uri, predict_options_uri)
    predictor.predict(scene_config_uri)
