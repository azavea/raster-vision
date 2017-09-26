import click

from rv.commands.predict import predict
from rv.commands.train import train
from rv.commands.predict_on_chips import predict_on_chips
from rv.commands.make_train_chips import make_train_chips
from rv.commands.make_tf_record import make_tf_record
from rv.commands.make_predict_chips import make_predict_chips
from rv.commands.filter_geojson import filter_geojson
from rv.commands.aggregate_predictions import aggregate_predictions


@click.group()
def run():
    pass


run.add_command(predict)
run.add_command(train)
run.add_command(predict_on_chips)
run.add_command(make_train_chips)
run.add_command(make_tf_record)
run.add_command(make_predict_chips)
run.add_command(filter_geojson)
run.add_command(aggregate_predictions)


if __name__ == '__main__':
    run()
