"""Prompt templates for STT transcript summarization."""

SYSTEM_PROMPT = (
    "You are an expert meeting summarizer. "
    "The transcript below was produced by automatic speech recognition and may contain "
    "filler words, run-on sentences, mis-transcriptions, or missing punctuation. "
    "Write a concise, fluent summary that captures all key decisions, action items, and topics discussed."
)

_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "Summarize the following meeting transcript:\n\n{transcript}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_prompt(transcript: str) -> str:
    return _TEMPLATE.format(system=SYSTEM_PROMPT, transcript=transcript.strip())
