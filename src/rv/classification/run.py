import click

from rv.classification.commands.prep_train_data import prep_train_data
from rv.classification.commands.get_image_stats import get_image_stats
from rv.classification.commands.split_dataset import split_dataset
from rv.classification.commands.train import train


@click.group()
def run():
    pass


run.add_command(prep_train_data)
run.add_command(get_image_stats)
run.add_command(split_dataset)
run.add_command(train)


if __name__ == '__main__':
    run()
