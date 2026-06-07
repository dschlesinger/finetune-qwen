"""
Scoring functions for STT summarization benchmarks.

Each function accepts lists of predictions and references (strings) and returns
a dict of metric_name -> float scores. All models are loaded lazily on first call.
"""

from __future__ import annotations

import functools
import pathlib
import sys

# UniEval is a script-style repo with no package setup; add it to the path.
_vendor = pathlib.Path(__file__).parents[4] / "vendor" / "UniEval"
if _vendor.exists() and str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L (unigram/bigram recall/precision/F1, returns F1)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k] += scores[k].fmeasure
    return {k: v / n for k, v in agg.items()}


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def bertscore_scores(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str | None = None,
) -> dict[str, float]:
    """BERTScore F1 (precision and recall also returned)."""
    from bert_score import score as _score

    kwargs: dict = dict(model_type=model_type, verbose=False)
    if device:
        kwargs["device"] = device
    P, R, F1 = _score(predictions, references, **kwargs)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


# ---------------------------------------------------------------------------
# AlignScore
# ---------------------------------------------------------------------------

@functools.cache
def _alignscorer(device: str = "cpu"):
    from alignscore import AlignScore
    return AlignScore(model="roberta-base", batch_size=8, device=device,
                      evaluation_mode="nli_sp")


def alignscore_scores(
    predictions: list[str],
    sources: list[str],
    device: str = "cpu",
) -> dict[str, float]:
    """AlignScore: faithfulness of each prediction against its source document.

    Args:
        predictions: Generated summaries.
        sources: Original transcripts (not reference summaries — AlignScore checks
                 factual consistency against the source, not a gold reference).
    """
    scorer = _alignscorer(device)
    scores = scorer.score(contexts=sources, claims=predictions)
    return {"alignscore": sum(scores) / len(scores)}


# ---------------------------------------------------------------------------
# UniEval
# ---------------------------------------------------------------------------

@functools.cache
def _unieval_scorer():
    from unieval.evaluator import get_evaluator
    return get_evaluator("summarization")


def unieval_scores(
    predictions: list[str],
    sources: list[str],
    references: list[str],
) -> dict[str, float]:
    """UniEval 4-dimensional summarization scores: coherence, consistency, fluency, relevance.

    Args:
        predictions: Generated summaries.
        sources: Original transcripts.
        references: Gold reference summaries.
    """
    from unieval.utils import convert_to_json

    data = convert_to_json(
        output_list=predictions,
        src_list=sources,
        ref_list=references,
    )
    evaluator = _unieval_scorer()
    results = evaluator.evaluate(data, print_result=False)

    keys = ["coherence", "consistency", "fluency", "relevance", "overall"]
    agg = {k: 0.0 for k in keys}
    for r in results:
        for k in keys:
            agg[k] += r.get(k, 0.0)
    n = len(results)
    return {f"unieval_{k}": v / n for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def score_all(
    predictions: list[str],
    references: list[str],
    sources: list[str] | None = None,
    device: str = "cpu",
    skip: list[str] | None = None,
) -> dict[str, float]:
    """Run all scorers and merge results into a single flat dict.

    Args:
        predictions: Model-generated summaries.
        references: Gold reference summaries.
        sources: Original transcripts. Required for AlignScore and UniEval.
                 If None, those two metrics are skipped.
        device: Torch device string (e.g. "cuda:0").
        skip: Metric names to skip, e.g. ["unieval", "alignscore"].
    """
    skip = set(skip or [])
    results: dict[str, float] = {}

    if "rouge" not in skip:
        results.update(rouge_scores(predictions, references))

    if "bertscore" not in skip:
        results.update(bertscore_scores(predictions, references, device=device))

    if sources is not None:
        if "alignscore" not in skip:
            results.update(alignscore_scores(predictions, sources, device=device))

        if "unieval" not in skip:
            results.update(unieval_scores(predictions, sources, references))

    return results
