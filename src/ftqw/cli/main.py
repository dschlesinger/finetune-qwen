import click
from ftqw import __version__
from ftqw.cli.download import download_data
from ftqw.cli.finetune import finetune
from ftqw.cli.benchmark import benchmark


@click.group()
@click.version_option(__version__, prog_name="ftqw")
def cli():
    """ftqw — Fine-tune Qwen for STT summarization."""


cli.add_command(download_data, name="download-data")
cli.add_command(finetune)
cli.add_command(benchmark)
