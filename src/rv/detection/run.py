import click

from rv.detection.commands.prep_train_data import prep_train_data
from rv.detection.commands.predict import predict
from rv.detection.commands.train import train
from rv.detection.commands.predict_on_chips import predict_on_chips
from rv.detection.commands.make_predict_chips import make_predict_chips
from rv.detection.commands.transform_geojson import transform_geojson
from rv.detection.commands.aggregate_predictions import aggregate_predictions
from rv.detection.commands.eval_predictions import eval_predictions
from rv.detection.commands.eval_model import eval_model
from rv.detection.commands.merge_predictions import merge_predictions
from rv.detection.commands.predict_array import predict_array


@click.group()
def run():
    pass


run.add_command(prep_train_data)
run.add_command(predict)
run.add_command(train)
run.add_command(predict_on_chips)
run.add_command(make_predict_chips)
run.add_command(transform_geojson)
run.add_command(aggregate_predictions)
run.add_command(eval_predictions)
run.add_command(eval_model)
run.add_command(merge_predictions)
run.add_command(predict_array)


if __name__ == '__main__':
    run()
