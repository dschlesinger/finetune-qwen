"""
Dataset loaders for AMI Meeting Corpus and MeetingBank.

Returns a normalized list[Meeting] regardless of source. Both benchmark scripts
and the fine-tuning pipeline import from here.

HuggingFace dataset IDs:
  AMI summaries:   TalTechNLP/AMIsum          field: id, transcript, summary
  AMI utterances:  edinburghcstr/ami           configs: ihm / sdm, field: meeting_id, text
  MeetingBank:     huuuyeah/MeetingBank        field: meeting_id, transcript, summary
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

    TalTechNLP/AMIsum provides meeting-level summaries and the official split.
    edinburghcstr/ami IHM/SDM configs provide utterance-level transcripts that
    are aggregated per meeting_id to reconstruct full-session text.

    IHM (individual headset mic) = cleaner near-manual transcripts → transcript_manual.
    SDM (single distant mic)     = noisier ASR-like transcripts    → transcript_asr.
    Falls back to the AMIsum transcript field if the config is unavailable.
    """
    ami_sum = load_dataset("TalTechNLP/AMIsum", split=split)

    ihm_by_id: dict[str, list[str]] = {}
    sdm_by_id: dict[str, list[str]] = {}
    try:
        for row in load_dataset("edinburghcstr/ami", "ihm", split=split):
            ihm_by_id.setdefault(row["meeting_id"], []).append(row["text"])
        for row in load_dataset("edinburghcstr/ami", "sdm", split=split):
            sdm_by_id.setdefault(row["meeting_id"], []).append(row["text"])
    except Exception:
        pass

    ihm_text = {mid: " ".join(utts) for mid, utts in ihm_by_id.items()}
    sdm_text = {mid: " ".join(utts) for mid, utts in sdm_by_id.items()}

    meetings: list[Meeting] = []
    for row in ami_sum:
        mid = row["id"]
        fallback = row["transcript"]
        meetings.append(
            Meeting(
                id=mid,
                source="ami",
                split=split,
                transcript_asr=sdm_text.get(mid, fallback),
                transcript_manual=ihm_text.get(mid, fallback),
                summary=row["summary"],
            )
        )
    return meetings


# ---------------------------------------------------------------------------
# MeetingBank
# ---------------------------------------------------------------------------

def load_meetingbank(split: str = "test") -> list[Meeting]:
    """
    Load MeetingBank (local government meetings, ASR transcripts only).

    MeetingBank has no manual transcripts, so transcript_manual is None.
    """
    ds = load_dataset("huuuyeah/MeetingBank", split=split)

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
