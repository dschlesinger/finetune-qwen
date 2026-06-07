from __future__ import annotations

import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ftqw.inference.prompt import format_chat

# LoRA target modules for Qwen2.5 attention + MLP projections
_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

_BNB_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_for_inference(model_path: str | pathlib.Path, adapter_path: str | pathlib.Path | None = None):
    """
    Load a local Qwen model in 4-bit for generation. Returns (model, tokenizer).

    Args:
        model_path:   Path to local base model weights.
        adapter_path: Optional path to a LoRA adapter saved by `ftqw finetune`.
                      When provided, the adapter is loaded on top of the base model.
    """
    model_path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=_BNB_4BIT,
        device_map="auto",
    )
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def load_for_training(model_path: str | pathlib.Path):
    """
    Load a local Qwen model in 4-bit with LoRA for QLoRA fine-tuning. Returns (model, tokenizer).

    Args:
        model_path: Path to local base model weights.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=_BNB_4BIT,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=_LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def generate_summary(model, tokenizer, transcript: str, max_new_tokens: int = 256) -> str:
    """Run greedy decoding on a single transcript and return the summary string."""
    messages = format_chat(transcript)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
