"""
Inference wrapper for Qwen2.5-7B-Instruct (base or fine-tuned).

Weights are NOT downloaded on import. Call model.load() explicitly when ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from inference.prompts import build_prompt

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True


class SummarizationModel:
    """
    Wraps Qwen2.5-7B-Instruct for transcript summarization.

    Parameters
    ----------
    adapter_path : str, optional
        Path to a fine-tuned LoRA adapter directory. If None, uses the base model.
    load_in_4bit : bool
        Enable bitsandbytes 4-bit quantization (recommended for 24 GB GPU).
    device_map : str
        Passed to from_pretrained; "auto" shards across available GPUs/CPU.
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "SummarizationModel":
        """Download (if needed) and load model weights into memory."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_cfg = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if self.load_in_4bit
            else None
        )

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_cfg,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16 if not self.load_in_4bit else None,
        )

        if self.adapter_path is not None:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)

        self._model.eval()
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def summarize(
        self,
        transcript: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Summarize a single STT transcript.

        Parameters
        ----------
        transcript : str
            Raw ASR or manual transcript text.
        config : GenerationConfig, optional
            Generation hyperparameters. Defaults are applied if omitted.

        Returns
        -------
        str
            The generated summary text (stripped of the prompt).
        """
        if not self.is_loaded:
            raise RuntimeError("Call model.load() before running inference.")

        if config is None:
            config = GenerationConfig()

        prompt = build_prompt(transcript)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        import torch
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def summarize_batch(
        self,
        transcripts: list[str],
        config: Optional[GenerationConfig] = None,
    ) -> list[str]:
        """Summarize a list of transcripts one at a time."""
        return [self.summarize(t, config) for t in transcripts]
