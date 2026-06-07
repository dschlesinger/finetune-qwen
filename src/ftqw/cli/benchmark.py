from __future__ import annotations

import json
import pathlib
from datetime import datetime

import click
from tqdm import tqdm


@click.command("benchmark")
@click.option("--model", required=True, metavar="PATH",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
              help="Local path to Qwen model weights.")
@click.option("--adapter", default=None, metavar="PATH",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
              help="Local path to LoRA adapter (output of ftqw finetune). Omit to benchmark the base model.")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=pathlib.Path),
              default="benchmarks/results", show_default=True)
@click.option("--max-new-tokens", default=256, show_default=True)
def benchmark(
    model: pathlib.Path,
    adapter: pathlib.Path | None,
    output_dir: pathlib.Path,
    max_new_tokens: int,
) -> None:
    """Benchmark a local Qwen model on the deterministic 20% held-out split."""
    from ftqw.data import load_all_splits, benchmark_split
    from ftqw.inference import load_for_inference, generate_summary
    from ftqw.benchmarks import score_all

    click.echo("Loading datasets...")
    meetings = benchmark_split(load_all_splits())
    click.echo(f"Benchmark split: {len(meetings)} meetings")

    click.echo(f"Loading model from {model}" + (f" + adapter {adapter}" if adapter else ""))
    mdl, tokenizer = load_for_inference(model, adapter_path=adapter)

    predictions, references, sources = [], [], []
    for meeting in tqdm(meetings, desc="Generating"):
        summary = generate_summary(mdl, tokenizer, meeting.transcript_asr, max_new_tokens)
        predictions.append(summary)
        references.append(meeting.summary)
        sources.append(meeting.transcript_asr)

    click.echo("Scoring...")
    scores = score_all(predictions, references, sources=sources)

    click.echo("\nResults:")
    for k, v in scores.items():
        click.echo(f"  {k}: {v:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    label = adapter.name if adapter else model.name
    out_file = output_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps({
        "model": str(model),
        "adapter": str(adapter) if adapter else None,
        "timestamp": datetime.now().isoformat(),
        "n_meetings": len(meetings),
        "scores": scores,
    }, indent=2))
    click.echo(f"\nSaved to {out_file}")
