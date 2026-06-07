"""
Dataset loaders for AMI Meeting Corpus and MeetingBank.

Returns a normalized list[Meeting] regardless of source. Both benchmark scripts
and the fine-tuning pipeline import from here.

HuggingFace dataset IDs (verify with `ds.features` if fields change):
  AMI:         edinburghcbid/ami   configs: "ihm" (manual) / "sdm" (distant-mic ASR)
  MeetingBank: huuuyeah/MeetingBank
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset

# Benchmark fraction — meetings whose SHA-256 bucket falls below this threshold
# are held out for benchmarking; the rest go to fine-tuning.
_BENCHMARK_PCT = 20  # out of 100


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


def load_all_splits() -> list[Meeting]:
    """Load every HuggingFace split (train + validation + test) for both datasets."""
    meetings: list[Meeting] = []
    for split in ("train", "validation", "test"):
        meetings.extend(load_ami(split))
        meetings.extend(load_meetingbank(split))
    return meetings


# ---------------------------------------------------------------------------
# Deterministic 80/20 split
# ---------------------------------------------------------------------------
# Assignment is a pure function of (source, meeting_id) via SHA-256 mod 100,
# so the same meeting always lands in the same bucket regardless of load order
# or which model is being evaluated.

def _bucket(meeting: Meeting) -> int:
    key = f"{meeting.source}:{meeting.id}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % 100


def finetune_split(meetings: list[Meeting]) -> list[Meeting]:
    """Return the ~80% of meetings reserved for fine-tuning."""
    return [m for m in meetings if _bucket(m) >= _BENCHMARK_PCT]


def benchmark_split(meetings: list[Meeting]) -> list[Meeting]:
    """Return the ~20% of meetings reserved for benchmarking."""
    return [m for m in meetings if _bucket(m) < _BENCHMARK_PCT]
