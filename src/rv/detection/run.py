import click

from rv.detection.commands.predict import predict
from rv.detection.commands.train import train
from rv.detection.commands.predict_on_chips import predict_on_chips
from rv.detection.commands.make_train_chips import make_train_chips
from rv.detection.commands.make_tf_record import make_tf_record
from rv.detection.commands.make_predict_chips import make_predict_chips
from rv.detection.commands.transform_geojson import transform_geojson
from rv.detection.commands.aggregate_predictions import aggregate_predictions
from rv.detection.commands.eval_predictions import eval_predictions
from rv.detection.commands.make_label_map import make_label_map
from rv.detection.commands.prep_train_data import prep_train_data
from rv.detection.commands.eval_model import eval_model


@click.group()
def run():
    pass


run.add_command(predict)
run.add_command(train)
run.add_command(predict_on_chips)
run.add_command(make_train_chips)
run.add_command(make_tf_record)
run.add_command(make_predict_chips)
run.add_command(transform_geojson)
run.add_command(aggregate_predictions)
run.add_command(eval_predictions)
run.add_command(make_label_map)
run.add_command(prep_train_data)
run.add_command(eval_model)


if __name__ == '__main__':
    run()
