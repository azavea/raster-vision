import click

from rv.cl.commands.prep_train_data import prep_train_data


@click.group()
def run():
    pass


run.add_command(prep_train_data)


if __name__ == '__main__':
    run()
