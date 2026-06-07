"""
Dataset loaders for AMI Meeting Corpus and MeetingBank.

Returns a normalized list[Meeting] regardless of source. Both benchmark scripts
and the fine-tuning pipeline import from here.

Data sources:
  AMI:         https://cs.taltech.ee/staff/heharm/AMIsum/  (train/val/test .json)
               Fields: id, transcript (manual IHM quality), summary
  MeetingBank: huuuyeah/MeetingBank (HF)
               Fields: meeting_id, transcript (ASR), summary

Note: edinburghcstr/ami is intentionally not used — it bundles 16kHz audio parquet
files making downloads multi-GB. AMIsum already contains high-quality IHM transcripts.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset

_AMISUM_BASE = "https://cs.taltech.ee/staff/heharm/AMIsum/"
_AMISUM_SPLIT_FILE = {"train": "train.json", "validation": "val.json", "test": "test.json"}

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

def _fetch_amisum(split: str) -> dict:
    """Download one AMIsum JSON split from the TalTech server."""
    filename = _AMISUM_SPLIT_FILE[split]
    with urllib.request.urlopen(_AMISUM_BASE + filename, timeout=30) as r:
        return json.load(r)


def load_ami(split: str = "test") -> list[Meeting]:
    """
    Load AMI Meeting Corpus via the TalTech AMIsum server.

    The AMIsum transcript is manual (IHM quality). Both transcript_asr and
    transcript_manual are set to the same value — the IHM/SDM distinction
    can be added later without touching callers.
    """
    raw = _fetch_amisum(split)
    meetings: list[Meeting] = []
    for mid, tx, sm in zip(raw["id"], raw["transcript"], raw["summary"]):
        meetings.append(
            Meeting(
                id=mid,
                source="ami",
                split=split,
                transcript_asr=tx,
                transcript_manual=tx,
                summary=sm,
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
