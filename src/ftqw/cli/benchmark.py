from __future__ import annotations

import json
import pathlib
from datetime import datetime

import click
from tqdm import tqdm

from ftqw.inference.model import DEFAULT_MODEL_ID


@click.command("benchmark")
@click.option("--model", default=DEFAULT_MODEL_ID, show_default=True, metavar="MODEL",
              help="HuggingFace model ID or local path to weights. Downloaded automatically if absent.")
@click.option("--local-dir", default=None, metavar="PATH",
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              help="Download weights here instead of the HF cache.")
@click.option("--adapter", default=None, metavar="PATH",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
              help="Local LoRA adapter (output of ftqw finetune). Omit to benchmark the base model.")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=pathlib.Path),
              default="benchmarks/results", show_default=True)
@click.option("--max-new-tokens", default=256, show_default=True)
def benchmark(
    model: str,
    local_dir: pathlib.Path | None,
    adapter: pathlib.Path | None,
    output_dir: pathlib.Path,
    max_new_tokens: int,
) -> None:
    """Benchmark a Qwen model on the deterministic 20% held-out split.

    By default runs the base model from HuggingFace, downloading weights on
    first use. Pass --adapter to evaluate a fine-tuned LoRA checkpoint instead.

    Examples:

    \b
      # Base model (auto-download to HF cache)
      ftqw benchmark

    \b
      # Base model, weights saved locally
      ftqw benchmark --local-dir models/Qwen2.5-7B-Instruct

    \b
      # Fine-tuned model
      ftqw benchmark --model models/Qwen2.5-7B-Instruct --adapter checkpoints/my-adapter
    """
    from ftqw.data import load_all_splits, benchmark_split
    from ftqw.inference import load_base_model, load_finetuned_model, generate_summary
    from ftqw.benchmarks import score_all

    click.echo("Loading datasets...")
    meetings = benchmark_split(load_all_splits())
    click.echo(f"Benchmark split: {len(meetings)} meetings")

    if adapter is not None:
        click.echo(f"Loading fine-tuned model: {model} + adapter {adapter}")
        mdl, tokenizer = load_finetuned_model(model, adapter)
        label = pathlib.Path(adapter).name
    else:
        click.echo(f"Loading base model: {model}")
        mdl, tokenizer = load_base_model(model, local_dir=local_dir)
        label = pathlib.Path(model).name if pathlib.Path(model).exists() else model.replace("/", "-")

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
    out_file = output_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps({
        "model": model,
        "adapter": str(adapter) if adapter else None,
        "timestamp": datetime.now().isoformat(),
        "n_meetings": len(meetings),
        "scores": scores,
    }, indent=2))
    click.echo(f"\nSaved to {out_file}")
