from __future__ import annotations

from ftqw.data.datasets import Meeting

SYSTEM_PROMPT = (
    "You are an expert at summarizing meeting transcripts. "
    "Produce a concise, accurate summary that captures key decisions, "
    "action items, and main discussion points."
)


def format_chat(transcript: str) -> list[dict]:
    """Return a messages list suitable for tokenizer.apply_chat_template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarize the following meeting transcript:\n\n{transcript}"},
    ]


def format_sft_example(meeting: Meeting, tokenizer) -> str:
    """Format a meeting as a complete chat string for SFT (includes assistant turn)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarize the following meeting transcript:\n\n{meeting.transcript_asr}"},
        {"role": "assistant", "content": meeting.summary},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)
