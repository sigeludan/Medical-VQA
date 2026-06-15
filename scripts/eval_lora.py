#!/usr/bin/env python3
"""Evaluate a LoRA-tuned InternVL2 checkpoint on VQA-RAD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_infer import load_image_tensor, load_jsonl, save_jsonl
from baseline_infer_v2 import PROMPT_VERSION, build_prompt
from internvl_sft_utils import load_model_with_lora
from vqa_common import normalize_text, summarize_metrics


@torch.inference_mode()
def run_lora_inference(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    records: list[dict],
    project_root: Path,
    max_num: int,
    max_new_tokens: int,
) -> list[dict]:
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    outputs: list[dict] = []

    for record in tqdm(records, desc="LoRA inference"):
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
    parser = argparse.ArgumentParser(description="Evaluate InternVL2 + LoRA on VQA-RAD.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Base InternVL2 model directory.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        required=True,
        help="LoRA adapter directory (saved by train_lora.ipynb).",
    )
    parser.add_argument(
        "--projector-path",
        type=Path,
        default=None,
        help="Optional mlp1 projector weights (mlp1_projector.pt from train_lora_mlp1.ipynb).",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="JSONL or JSON list file. Defaults to val_internvl.json fields mapped from val jsonl.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Shortcut: evaluate vqa_rad_val.jsonl or vqa_rad_test.jsonl directly.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--predictions-file",
        type=str,
        default="lora_predictions.jsonl",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="lora_metrics.json",
    )
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument(
        "--relaxed-metrics",
        action="store_true",
        help="Also report closed_relaxed_match (synonyms + word-boundary containment).",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    model_path = (args.model_path or project_root / "models/OpenGVLab/InternVL2-8B").resolve()
    lora_path = args.lora_path.resolve()
    projector_path = args.projector_path.resolve() if args.projector_path else None
    output_dir = (args.output_dir or project_root / "outputs").resolve()

    if args.test_file is not None:
        records = load_jsonl(args.test_file.resolve())
    elif args.eval_file is not None:
        eval_path = args.eval_file.resolve()
        if eval_path.suffix == ".jsonl":
            records = load_jsonl(eval_path)
        else:
            records = json.loads(eval_path.read_text(encoding="utf-8"))
            records = [
                {
                    "image": row["image"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "answer_type": row.get("answer_type", "closed"),
                    "split": row.get("split", "val"),
                }
                for row in records
            ]
    else:
        records = load_jsonl(project_root / "data/vqa_rad_val.jsonl")

    if args.limit > 0:
        records = records[: args.limit]

    print(f"Base model: {model_path}")
    print(f"LoRA adapter: {lora_path}")
    if projector_path is not None:
        print(f"Projector weights: {projector_path}")
    print(f"Evaluating {len(records)} samples")

    model, tokenizer = load_model_with_lora(
        model_path=model_path,
        lora_path=lora_path,
        projector_path=projector_path,
        use_flash_attn=args.use_flash_attn,
    )
    predictions = run_lora_inference(
        model=model,
        tokenizer=tokenizer,
        records=records,
        project_root=project_root,
        max_num=args.max_num,
        max_new_tokens=args.max_new_tokens,
    )

    metrics = summarize_metrics(predictions, include_relaxed=args.relaxed_metrics)
    metrics.update(
        {
            "prompt_version": PROMPT_VERSION,
            "model_path": str(model_path),
            "lora_path": str(lora_path),
            "projector_path": str(projector_path) if projector_path else None,
            "num_samples": len(records),
            "limit": args.limit,
        }
    )

    pred_path = output_dir / args.predictions_file
    metrics_path = output_dir / args.metrics_file
    save_jsonl(pred_path, predictions)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nLoRA metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved predictions -> {pred_path}")
    print(f"Saved metrics      -> {metrics_path}")


if __name__ == "__main__":
    main()
