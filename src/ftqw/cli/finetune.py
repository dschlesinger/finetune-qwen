from __future__ import annotations

import pathlib

import click


@click.command("finetune")
@click.option("--model", required=True, metavar="PATH",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
              help="Local path to base Qwen model weights.")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=pathlib.Path),
              default="checkpoints", show_default=True,
              help="Directory to save the LoRA adapter.")
@click.option("--epochs", default=3, show_default=True)
@click.option("--lr", default=2e-4, show_default=True, help="Learning rate.")
@click.option("--batch-size", default=4, show_default=True,
              help="Per-device training batch size.")
@click.option("--grad-accum", default=4, show_default=True,
              help="Gradient accumulation steps (effective batch = batch-size × grad-accum).")
@click.option("--max-seq-len", default=4096, show_default=True,
              help="Max token length per training example.")
def finetune(
    model: pathlib.Path,
    output_dir: pathlib.Path,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    max_seq_len: int,
) -> None:
    """QLoRA fine-tune a local Qwen model on the deterministic 80% split."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from ftqw.data import load_all_splits, finetune_split
    from ftqw.inference import load_for_training, format_sft_example

    click.echo("Loading datasets...")
    meetings = finetune_split(load_all_splits())
    click.echo(f"Fine-tune split: {len(meetings)} meetings")

    click.echo(f"Loading model from {model}")
    mdl, tokenizer = load_for_training(model)

    click.echo("Formatting training examples...")
    hf_dataset = Dataset.from_dict({
        "text": [format_sft_example(m, tokenizer) for m in meetings]
    })

    save_path = output_dir / model.name
    save_path.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=mdl,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        args=SFTConfig(
            output_dir=str(save_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            max_seq_length=max_seq_len,
            dataset_text_field="text",
            report_to="none",
        ),
    )

    click.echo("Training...")
    trainer.train()
    trainer.save_model(str(save_path))
    click.echo(f"Adapter saved to {save_path}")
