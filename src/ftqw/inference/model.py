from __future__ import annotations

import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ftqw.inference.prompt import format_chat

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

_BNB_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ---------------------------------------------------------------------------
# Base model (with auto-download)
# ---------------------------------------------------------------------------

def load_base_model(
    model_id: str = DEFAULT_MODEL_ID,
    local_dir: str | pathlib.Path | None = None,
):
    """
    Load the base Qwen model in 4-bit NF4 for inference. Downloads weights if needed.

    Args:
        model_id:  HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B-Instruct") or a
                   local path to already-downloaded weights.
        local_dir: If given, weights are downloaded here when absent, then loaded
                   from disk. Useful for keeping weights outside the HF cache.
                   Ignored when model_id is already a local path that exists.
    """
    source = _resolve_model_source(model_id, local_dir)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=_BNB_4BIT,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _resolve_model_source(model_id: str, local_dir: str | pathlib.Path | None) -> str:
    """Return the string path or HF ID to pass to from_pretrained, downloading if needed."""
    # If model_id is already a local directory with weights, use it directly.
    local_candidate = pathlib.Path(model_id)
    if local_candidate.is_dir() and (local_candidate / "config.json").exists():
        return str(local_candidate)

    # If a local_dir was requested, download there if the weights aren't present.
    if local_dir is not None:
        local_dir = pathlib.Path(local_dir)
        if not (local_dir / "config.json").exists():
            _download_weights(model_id, local_dir)
        return str(local_dir)

    # Fall back to HuggingFace cache (from_pretrained handles download automatically).
    return model_id


def _download_weights(model_id: str, local_dir: pathlib.Path) -> None:
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_id} → {local_dir} ...")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    print("Download complete.")


# ---------------------------------------------------------------------------
# Fine-tuned model (base + LoRA adapter, both local)
# ---------------------------------------------------------------------------

def load_finetuned_model(
    base_path: str | pathlib.Path,
    adapter_path: str | pathlib.Path,
):
    """
    Load a base model from disk and apply a LoRA adapter on top.

    Args:
        base_path:    Local path to the base model weights.
        adapter_path: Local path to the LoRA adapter saved by `ftqw finetune`.
    """
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        quantization_config=_BNB_4BIT,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training loader (QLoRA)
# ---------------------------------------------------------------------------

def load_for_training(model_path: str | pathlib.Path):
    """Load in 4-bit with LoRA attached for QLoRA fine-tuning. Returns (model, tokenizer)."""
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
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=_LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_summary(model, tokenizer, transcript: str, max_new_tokens: int = 256) -> str:
    """Run greedy decoding on a single transcript and return the summary string."""
    messages = format_chat(transcript)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# GGUF backend (llama-cpp-python)
# ---------------------------------------------------------------------------

def load_gguf_for_inference(
    model_path: str | pathlib.Path,
    n_gpu_layers: int = -1,
    n_ctx: int = 8192,
):
    """
    Load a GGUF model via llama-cpp-python. Returns a Llama instance.

    Args:
        model_path:   Path to a .gguf file.
        n_gpu_layers: Layers to offload to GPU (-1 = all).
        n_ctx:        Context window size in tokens.
    """
    from llama_cpp import Llama

    return Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )


def generate_summary_gguf(llm, transcript: str, max_new_tokens: int = 256) -> str:
    """Run greedy chat completion on a GGUF model and return the summary string."""
    messages = format_chat(transcript)
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"].strip()
