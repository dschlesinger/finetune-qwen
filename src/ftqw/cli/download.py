import pathlib

import click
from datasets import load_dataset


@click.command()
@click.option(
    "--dataset",
    "dataset_name",
    type=click.Choice(["ami", "meetingbank", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Which dataset(s) to download.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    default="data/raw",
    show_default=True,
    help="Directory to save datasets into.",
)
def download_data(dataset_name: str, output_dir: pathlib.Path) -> None:
    """Download AMI and/or MeetingBank from HuggingFace and save to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = ["ami", "meetingbank"] if dataset_name == "all" else [dataset_name]
    for name in targets:
        if name == "ami":
            _download_ami(output_dir)
        else:
            _download_meetingbank(output_dir)


def _download_ami(output_dir: pathlib.Path) -> None:
    # Meeting-level summaries (TalTechNLP/AMIsum)
    for split in ("train", "validation", "test"):
        click.echo(f"[ami/sum] downloading split={split} ...")
        ds = load_dataset("TalTechNLP/AMIsum", split=split)
        dest = output_dir / "ami" / "sum" / split
        ds.save_to_disk(str(dest))
        click.echo(f"  -> {len(ds)} rows saved to {dest}")

    # Utterance-level transcripts (edinburghcstr/ami, IHM + SDM)
    for config in ("ihm", "sdm"):
        for split in ("train", "validation", "test"):
            click.echo(f"[ami/{config}] downloading split={split} ...")
            try:
                ds = load_dataset("edinburghcstr/ami", config, split=split)
                dest = output_dir / "ami" / config / split
                ds.save_to_disk(str(dest))
                click.echo(f"  -> {len(ds)} rows saved to {dest}")
            except Exception as exc:
                click.echo(f"  [warn] skipped: {exc}")


def _download_meetingbank(output_dir: pathlib.Path) -> None:
    for split in ("train", "validation", "test"):
        click.echo(f"[meetingbank] downloading split={split} ...")
        ds = load_dataset("huuuyeah/MeetingBank", split=split)
        dest = output_dir / "meetingbank" / split
        ds.save_to_disk(str(dest))
        click.echo(f"  -> {len(ds)} rows saved to {dest}")
