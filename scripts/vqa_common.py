"""Shared helpers for VQA-RAD preprocessing and evaluation."""

from __future__ import annotations

import re
from typing import Iterable

# Normalized synonym groups for VQA-RAD short answers (all strings lowercase).
SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"axial", "axial plane"}),
    frozenset({"brain", "the brain"}),
    frozenset({"4th ventricle", "fourth ventricle", "4 ventricle"}),
    frozenset({"csf", "cerebrospinal fluid"}),
    frozenset({"x ray", "xray", "chest x ray", "cxr"}),
    frozenset({"mri", "mri flair", "flair", "mri-flair"}),
    frozenset({"left", "left side"}),
    frozenset({"right", "right side"}),
    frozenset({"posterior", "posteriorly"}),
    frozenset({"anterior", "anteriorly"}),
    frozenset({"non contrast", "noncontrast", "non contrast ct"}),
    frozenset({"contrast", "with contrast", "contrast ct"}),
    frozenset({"ct", "ct scan"}),
    frozenset({"hypodense", "hypodense lesion"}),
    frozenset({"hyperdense", "hyperdense lesion"}),
    frozenset({"enlarged", "enlargement"}),
    frozenset({"pleural effusion", "effusion"}),
    frozenset({"cardiomegaly", "enlarged heart"}),
)

_SYNONYM_LOOKUP: dict[str, frozenset[str]] = {}
for group in SYNONYM_GROUPS:
    for term in group:
        _SYNONYM_LOOKUP[term] = group


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace(" ?", "?")
    text = re.sub(r"[^\w\s\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_answer_type(question: str, answer: str) -> str:
    """Heuristic closed vs open labeling for VQA-RAD."""
    ans = normalize_text(answer)
    if ans in {"yes", "no"}:
        return "closed"

    q = normalize_text(question)
    closed_prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "does ",
        "do ",
        "can ",
        "could ",
        "has ",
        "have ",
        "will ",
        "did ",
        "is there ",
        "are there ",
        "is this ",
        "is the ",
        "are the ",
    )
    if any(q.startswith(prefix) for prefix in closed_prefixes) and len(ans.split()) <= 4:
        return "closed"
    if len(ans.split()) <= 2:
        return "closed"
    return "open"


def extract_yes_no(text: str) -> str | None:
    text = normalize_text(text)
    first = text.split()[0] if text.split() else ""
    if first in {"yes", "no"}:
        return first
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def exact_match(pred: str, ref: str) -> bool:
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    if pred_norm == ref_norm:
        return True
    if ref_norm in {"yes", "no"}:
        pred_yes_no = extract_yes_no(pred_norm)
        return pred_yes_no == ref_norm
    return False


def synonyms_match(pred: str, ref: str) -> bool:
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    pred_group = _SYNONYM_LOOKUP.get(pred_norm)
    ref_group = _SYNONYM_LOOKUP.get(ref_norm)
    if pred_group is None or ref_group is None:
        return False
    return pred_group == ref_group


def word_containment_match(pred: str, ref: str, min_len: int = 3) -> bool:
    """True when the shorter answer appears as a whole-phrase substring of the longer."""
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    if not pred_norm or not ref_norm:
        return False
    shorter, longer = (pred_norm, ref_norm) if len(pred_norm) <= len(ref_norm) else (ref_norm, pred_norm)
    if len(shorter) < min_len:
        return False
    if shorter == longer:
        return True
    # Avoid e.g. normal in abnormal: require word-boundary containment.
    padded = f" {longer} "
    needle = f" {shorter} "
    return needle in padded


def relaxed_match(pred: str, ref: str) -> bool:
    if exact_match(pred, ref):
        return True
    ref_norm = normalize_text(ref)
    if ref_norm in {"yes", "no"}:
        return False
    if synonyms_match(pred, ref):
        return True
    return word_containment_match(pred, ref)


def closed_match(pred: str, ref: str, *, relaxed: bool = False) -> bool:
    if relaxed:
        return relaxed_match(pred, ref)
    return exact_match(pred, ref)


def tokenize_for_bleu(text: str) -> list[str]:
    return normalize_text(text).split()


def bleu1(pred: str, ref: str) -> float:
    pred_tokens = tokenize_for_bleu(pred)
    ref_tokens = tokenize_for_bleu(ref)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    matches = sum(1 for token in pred_tokens if token in ref_tokens)
    precision = matches / len(pred_tokens)
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    if pred_len == 0:
        return 0.0
    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = pow(2.718281828, 1 - ref_len / pred_len) if pred_len else 0.0
    return bp * precision


def rouge_l(pred: str, ref: str) -> float:
    pred_tokens = tokenize_for_bleu(pred)
    ref_tokens = tokenize_for_bleu(ref)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    rows = len(pred_tokens) + 1
    cols = len(ref_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def summarize_metrics(
    records: Iterable[dict],
    *,
    include_relaxed: bool = False,
) -> dict:
    records = list(records)
    closed = [r for r in records if r.get("answer_type") == "closed"]
    opened = [r for r in records if r.get("answer_type") == "open"]

    summary = {
        "num_samples": len(records),
        "num_closed": len(closed),
        "num_open": len(opened),
    }
    if closed:
        summary["closed_exact_match"] = sum(
            1 for r in closed if exact_match(r["prediction"], r["answer"])
        ) / len(closed)
        if include_relaxed:
            summary["closed_relaxed_match"] = sum(
                1 for r in closed if relaxed_match(r["prediction"], r["answer"])
            ) / len(closed)
    if opened:
        summary["open_bleu1"] = sum(
            bleu1(r["prediction"], r["answer"]) for r in opened
        ) / len(opened)
        summary["open_rouge_l"] = sum(
            rouge_l(r["prediction"], r["answer"]) for r in opened
        ) / len(opened)
    return summary
