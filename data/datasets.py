"""
Dataset loaders for AMI Meeting Corpus and MeetingBank.

Returns a normalized list[Meeting] regardless of source. Both benchmark scripts
and the fine-tuning pipeline import from here.

HuggingFace dataset IDs (verify with `ds.features` if fields change):
  AMI:         edinburghcbid/ami   configs: "ihm" (manual) / "sdm" (distant-mic ASR)
  MeetingBank: huuuyeah/MeetingBank
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset


@dataclass
class Meeting:
    id: str
    source: str                    # "ami" | "meetingbank"
    split: str                     # "train" | "validation" | "test"
    transcript_asr: str
    transcript_manual: Optional[str]  # None when unavailable
    summary: str


# ---------------------------------------------------------------------------
# AMI Meeting Corpus
# ---------------------------------------------------------------------------

def load_ami(split: str = "test") -> list[Meeting]:
    """
    Load AMI Meeting Corpus.

    IHM (individual headset mic) = high-quality near-manual transcripts.
    SDM (single distant mic)     = noisier, closer to real-world ASR output.

    Both configs share the same meeting IDs, so we zip them by meeting_id to
    produce one Meeting per session with both transcript variants.
    """
    ds_ihm = load_dataset(
        "edinburghcbid/ami", "ihm", split=split, trust_remote_code=True
    )
    ds_sdm = load_dataset(
        "edinburghcbid/ami", "sdm", split=split, trust_remote_code=True
    )

    sdm_by_id: dict[str, str] = {
        row["meeting_id"]: _ami_concat(row) for row in ds_sdm
    }

    meetings: list[Meeting] = []
    for row in ds_ihm:
        mid = row["meeting_id"]
        meetings.append(
            Meeting(
                id=mid,
                source="ami",
                split=split,
                transcript_asr=sdm_by_id.get(mid, ""),
                transcript_manual=_ami_concat(row),
                summary=row["abstractive_summary"],
            )
        )
    return meetings


def _ami_concat(row: dict) -> str:
    """Flatten AMI word-level transcript list to a plain string."""
    words = row.get("words", [])
    if isinstance(words, list):
        return " ".join(
            w["word"] if isinstance(w, dict) else str(w) for w in words
        )
    return str(words)


# ---------------------------------------------------------------------------
# MeetingBank
# ---------------------------------------------------------------------------

def load_meetingbank(split: str = "test") -> list[Meeting]:
    """
    Load MeetingBank (local government meetings, ASR transcripts only).

    MeetingBank has no manual transcripts, so transcript_manual is None.
    """
    ds = load_dataset(
        "huuuyeah/MeetingBank", split=split, trust_remote_code=True
    )

    meetings: list[Meeting] = []
    for row in ds:
        meetings.append(
            Meeting(
                id=row["meeting_id"],
                source="meetingbank",
                split=split,
                transcript_asr=row["transcript"],
                transcript_manual=None,
                summary=row["summary"],
            )
        )
    return meetings


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all(split: str = "test") -> list[Meeting]:
    """Load AMI + MeetingBank combined for the given split."""
    return load_ami(split) + load_meetingbank(split)
