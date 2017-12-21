import click

from rv.classification.commands.prep_train_data import prep_train_data
from rv.classification.commands.train import train


@click.group()
def run():
    pass


run.add_command(prep_train_data)
run.add_command(train)


if __name__ == '__main__':
    run()
