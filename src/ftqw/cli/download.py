import json
import pathlib
import urllib.request

import click
from datasets import load_dataset  # used by _download_meetingbank

_AMISUM_BASE = "https://cs.taltech.ee/staff/heharm/AMIsum/"
_AMISUM_FILES = {"train": "train.json", "validation": "val.json", "test": "test.json"}


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
    # AMIsum — plain JSON from TalTech server (no audio, text-only)
    for split, filename in _AMISUM_FILES.items():
        dest_dir = output_dir / "ami"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        if dest.exists():
            click.echo(f"[ami] {split} already downloaded, skipping.")
            continue
        click.echo(f"[ami] downloading split={split} ...")
        url = _AMISUM_BASE + filename
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        dest.write_text(json.dumps(data))
        click.echo(f"  -> {len(data['id'])} meetings saved to {dest}")


def _download_meetingbank(output_dir: pathlib.Path) -> None:
    for split in ("train", "validation", "test"):
        dest = output_dir / "meetingbank" / split
        if dest.exists() and any(dest.iterdir()):
            click.echo(f"[meetingbank] {split} already downloaded, skipping.")
            continue
        click.echo(f"[meetingbank] downloading split={split} ...")
        ds = load_dataset("huuuyeah/MeetingBank", split=split)
        ds.save_to_disk(str(dest))
        click.echo(f"  -> {len(ds)} rows saved to {dest}")
