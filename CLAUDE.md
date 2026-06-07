# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Fine-tune a Qwen model to improve summarization of speech-to-text (STT) transcripts. The project includes:
1. A general summarization benchmark (ROUGE, BERTScore against reference summaries)
2. A domain-specific benchmark for handling messy STT artifacts — run-on sentences, filler words ("um", "uh"), mis-transcriptions, speaker diarization noise, missing punctuation

## Package

Installable as `finetune-qwen` via:
```bash
pip install -e .
```

CLI entry point: `ftqw`

## Source Layout

```
src/ftqw/
  cli/
    main.py        # Click group root → ftqw
    download.py    # ftqw download-data
    benchmark.py   # stub group
    train.py       # stub group
  data/
    datasets.py    # Meeting dataclass + load_ami(), load_meetingbank(), load_all()
  benchmarks/
    scoring.py     # rouge_scores(), bertscore_scores(), alignscore_scores(), unieval_scores(), score_all()
data/              # runtime directory — downloaded datasets land here (not a Python package)
  raw/             # created by ftqw download-data
  processed/       # tokenized / formatted for training
```

## Key Design Decisions

- **Model**: Qwen (API-based or weights depending on access tier)
- **Training approach**: Supervised fine-tuning (SFT) on (messy-transcript, clean-summary) pairs
- **Datasets**: AMI Meeting Corpus + MeetingBank (both have real ASR transcripts + reference summaries)
- **Benchmark 1 – General summarization**: Evaluate summary quality on held-out transcripts
- **Benchmark 2 – STT robustness**: Compare quality on ASR vs. manual transcripts; degradation delta is the headline metric

## Benchmark Metric Stack

All metrics run locally, no external API required:

| Metric | Library | Purpose |
|---|---|---|
| **AlignScore** | `alignscore` | Faithfulness/consistency of summary vs. source transcript |
| **UniEval** | `unieval` (`MingZhong/unieval-sum` on HuggingFace) | Coherence, fluency, relevance (4-dimensional, T5-large based) |
| **ROUGE-1/2/L** | `rouge-score` | N-gram overlap baseline |
| **BERTScore** | `bert-score` | Semantic similarity to reference summary |

## Imports

```python
from ftqw.data import Meeting, load_ami, load_meetingbank, load_all
from ftqw.benchmarks import score_all
```

## Development Commands

```bash
# Install package in editable mode (run once)
pip install -e .

# Download datasets to data/raw/
ftqw download-data --dataset all

# (coming) Run general summarization benchmark
ftqw benchmark run --model <checkpoint>

# (coming) Run fine-tuning
ftqw train sft --config training/config.yaml
```
