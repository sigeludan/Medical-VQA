#!/usr/bin/env python3
"""Zero-shot baseline inference (prompt v2) for Medical VQA with InternVL2."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_infer import (
    load_image_tensor,
    load_jsonl,
    load_model,
    save_jsonl,
)
from vqa_common import normalize_text, summarize_metrics

import torch
from transformers import AutoModel, AutoTokenizer

PROMPT_VERSION = "v2"
DEFAULT_PRED_FILE = "baseline_v2_predictions.jsonl"
DEFAULT_METRICS_FILE = "baseline_v2_metrics.json"


def is_yes_no_question(question: str) -> bool:
    """Heuristic: only true binary questions should get a yes/no style answer."""
    q = question.strip().lower()
    q = re.sub(r"\s+", " ", q)

    if q.startswith(("what ", "where ", "how ", "which ", "who ", "when ", "from what ")):
        return False
    if " or " in q and q.startswith(("is ", "are ", "was ", "were ")):
        # e.g. "is this an mri or a ct scan?" -> short phrase, not yes/no
        return False

    yes_no_prefixes = (
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
        "is the ",
        "are the ",
        "is this ",
        "is it ",
        "can you see ",
        "can you evaluate ",
    )
    return any(q.startswith(prefix) for prefix in yes_no_prefixes)


def build_prompt(question: str) -> str:
    question = question.strip()

    # --- prompt v1 (over-generalized yes/no; kept for reference) ---
    # return (
    #     "You are a radiology expert. Look at the medical image and answer the "
    #     "question as briefly as possible. For yes/no questions, reply with only "
    #     f"'yes' or 'no'.\nQuestion: {question}"
    # )

    if is_yes_no_question(question):
        return (
            "You are a radiology expert. This is a yes/no question about a medical "
            "image. Reply with only one word: 'yes' or 'no'.\n"
            f"Question: {question}"
        )

    return (
        "You are a radiology expert. Answer the following question about the "
        "medical image using a very short phrase (1-6 words), such as an organ "
        "name, imaging plane, modality, side (left/right), or location. "
        "Do not reply with only 'yes' or 'no'.\n"
        f"Question: {question}"
    )


@torch.inference_mode()
def run_inference(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    records: list[dict],
    project_root: Path,
    max_num: int,
    max_new_tokens: int,
) -> list[dict]:
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    outputs: list[dict] = []

    for record in tqdm(records, desc="Baseline inference v2"):
        image_path = project_root / record["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pixel_values = load_image_tensor(image_path, max_num=max_num).to(torch.bfloat16).cuda()
        question = build_prompt(record["question"])
        prediction = model.chat(tokenizer, pixel_values, question, generation_config)
        prediction = normalize_text(prediction)

        outputs.append(
            {
                **record,
                "prediction": prediction,
                "prompt": question,
                "prompt_version": PROMPT_VERSION,
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run InternVL2 zero-shot baseline (prompt v2) on VQA-RAD."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to InternVL2 checkpoint.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="JSONL file for evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of test samples to evaluate. Use 0 for full test set.",
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=6,
        help="Max image tiles for InternVL dynamic preprocessing.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per answer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=DEFAULT_PRED_FILE,
        help="Predictions JSONL filename inside output-dir.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=DEFAULT_METRICS_FILE,
        help="Metrics JSON filename inside output-dir.",
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Enable flash attention if installed.",
    )
    parser.add_argument(
        "--relaxed-metrics",
        action="store_true",
        help="Also report closed_relaxed_match (synonyms + word-boundary containment).",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    model_path = (args.model_path or project_root / "models/OpenGVLab/InternVL2-8B").resolve()
    test_file = (args.test_file or project_root / "data/vqa_rad_test.jsonl").resolve()
    output_dir = (args.output_dir or project_root / "outputs").resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}. Run preprocess_vqa_rad.py first."
        )

    records = load_jsonl(test_file)
    if args.limit > 0:
        records = records[: args.limit]

    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Model: {model_path}")
    print(f"Evaluating {len(records)} samples from {test_file}")

    use_flash_attn = args.use_flash_attn
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("flash_attn not installed, falling back to standard attention.")
            use_flash_attn = False

    model, tokenizer = load_model(model_path, use_flash_attn=use_flash_attn)
    predictions = run_inference(
        model=model,
        tokenizer=tokenizer,
        records=records,
        project_root=project_root,
        max_num=args.max_num,
        max_new_tokens=args.max_new_tokens,
    )

    metrics = summarize_metrics(predictions, include_relaxed=args.relaxed_metrics)
    metrics["prompt_version"] = PROMPT_VERSION
    metrics["model_path"] = str(model_path)
    metrics["test_file"] = str(test_file)
    metrics["limit"] = args.limit

    pred_path = output_dir / args.predictions_file
    metrics_path = output_dir / args.metrics_file
    save_jsonl(pred_path, predictions)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nBaseline metrics (prompt v2):")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved predictions -> {pred_path}")
    print(f"Saved metrics      -> {metrics_path}")


if __name__ == "__main__":
    main()
