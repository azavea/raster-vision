import click

from abc import ABC, abstractmethod


class Command(ABC):
    @abstractmethod
    def run(self, tmp_dir: str, dry_run:bool=False):
        """Run the command."""
        pass

    @staticmethod
    def announce_dry_run():
        click.echo(click.style('dryrun: ', fg='blue'), nl=False)


class NoOpCommand(Command):
    """Defines a command that does nothing.
    """

    def run(self, tmp_dir: str, dry_run: bool=False):
        pass
