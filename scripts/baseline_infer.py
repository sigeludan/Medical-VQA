#!/usr/bin/env python3
"""Zero-shot baseline inference for Medical VQA with InternVL2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vqa_common import normalize_text, summarize_metrics

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: set[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image_tensor(image_file: Path, input_size: int = 448, max_num: int = 6) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(question: str) -> str:
    question = question.strip()
    return (
        "You are a radiology expert. Look at the medical image and answer the "
        "question as briefly as possible. For yes/no questions, reply with only "
        f"'yes' or 'no'.\nQuestion: {question}"
    )


def load_model(model_path: Path, use_flash_attn: bool) -> tuple[AutoModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if use_flash_attn:
        model_kwargs["use_flash_attn"] = True
    model = AutoModel.from_pretrained(model_path, **model_kwargs).eval().cuda()
    return model, tokenizer


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

    for record in tqdm(records, desc="Baseline inference"):
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
            }
        )
    return outputs


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InternVL2 zero-shot baseline on VQA-RAD.")
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
        "--use-flash-attn",
        action="store_true",
        help="Enable flash attention if installed.",
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

    metrics = summarize_metrics(predictions)
    metrics["model_path"] = str(model_path)
    metrics["test_file"] = str(test_file)
    metrics["limit"] = args.limit

    pred_path = output_dir / "baseline_predictions.jsonl"
    metrics_path = output_dir / "baseline_metrics.json"
    save_jsonl(pred_path, predictions)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nBaseline metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved predictions -> {pred_path}")
    print(f"Saved metrics      -> {metrics_path}")


if __name__ == "__main__":
    main()
