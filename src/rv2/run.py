import click

from rv2.builders import (
    make_train_data_builder, train_builder, predict_builder)


@click.command()
@click.argument('config_uri')
def make_train_data(config_uri):
    command = make_train_data_builder.build(config_uri)
    command.run()


@click.command()
@click.argument('config_uri')
def train(config_uri):
    command = train_builder.build(config_uri)
    command.run()


@click.command()
@click.argument('config_uri')
def predict(config_uri):
    command = predict_builder.build(config_uri)
    command.run()


@click.group()
def run():
    pass


run.add_command(make_train_data)
run.add_command(train)
run.add_command(predict)


if __name__ == '__main__':
    run()
