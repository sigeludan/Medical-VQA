#!/usr/bin/env python3
"""
Preprocess VQA-RAD into JSONL files for SFT / evaluation.

Output format (one JSON object per line):
  {"image": "...", "question": "...", "answer": "...", "answer_type": "closed|open"}

Split policy:
  - 7:1:2 train / val / test
  - Samples sharing the same image always stay in the same split
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from vqa_common import classify_answer_type, normalize_text


def load_records_from_parquet(raw_dir: Path, image_dir: Path, project_root: Path) -> list[dict]:
    records: list[dict] = []
    parquet_files = {
        "train": raw_dir / "data/train-00000-of-00001-eb8844602202be60.parquet",
        "test": raw_dir / "data/test-00000-of-00001-e5bc3d208bb4deeb.parquet",
    }

    for split_name, parquet_path in parquet_files.items():
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

        table = pq.read_table(parquet_path)
        for idx, row in enumerate(table.to_pylist()):
            image_info = row["image"]
            image_bytes = image_info.get("bytes")
            src_path = image_info.get("path") or f"{split_name}_{idx}.jpg"
            suffix = Path(src_path).suffix or ".jpg"
            image_name = f"{split_name}_{idx:04d}{suffix}"
            out_path = image_dir / image_name

            if image_bytes and not out_path.exists():
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.open(BytesIO(image_bytes)).convert("RGB").save(out_path)

            rel_image = out_path.relative_to(project_root).as_posix()
            question = normalize_text(row["question"])
            answer = normalize_text(row["answer"])
            records.append(
                {
                    "image": rel_image,
                    "question": question,
                    "answer": answer,
                    "answer_type": classify_answer_type(question, answer),
                    "source_split": split_name,
                }
            )
    return records


def load_records_from_jsonl(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            question = normalize_text(row["question"])
            answer = normalize_text(row["answer"])
            records.append(
                {
                    "image": row["image"],
                    "question": question,
                    "answer": answer,
                    "answer_type": classify_answer_type(question, answer),
                    "source_split": row.get("split", "unknown"),
                }
            )
    return records


def split_by_image(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["image"]].append(record)

    image_keys = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(image_keys)

    total = len(image_keys)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = set(image_keys[:train_end])
    val_images = set(image_keys[train_end:val_end])
    test_images = set(image_keys[val_end:])

    def collect(image_set: set[str]) -> list[dict]:
        split_records: list[dict] = []
        for image in sorted(image_set):
            split_records.extend(grouped[image])
        return split_records

    return collect(train_images), collect(val_images), collect(test_images)


def write_jsonl(path: Path, records: list[dict], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            item = {
                "image": record["image"],
                "question": record["question"],
                "answer": record["answer"],
                "answer_type": record["answer_type"],
                "split": split_name,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess VQA-RAD into train/val/test JSONL.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing HF parquet files.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional existing JSONL to preprocess instead of parquet.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    raw_dir = (args.raw_dir or project_root / "data/vqa-rad-raw").resolve()
    image_dir = project_root / "data/images"
    input_jsonl = args.input_jsonl or project_root / "data/vqa_rad_all.jsonl"

    if args.input_jsonl is not None or (
        not (raw_dir / "data/train-00000-of-00001-eb8844602202be60.parquet").exists()
        and input_jsonl.exists()
    ):
        print(f"Loading records from JSONL: {input_jsonl}")
        records = load_records_from_jsonl(input_jsonl)
    else:
        print(f"Loading records from parquet: {raw_dir}")
        records = load_records_from_parquet(raw_dir, image_dir, project_root)

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    train_records, val_records, test_records = split_by_image(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    out_dir = project_root / "data"
    write_jsonl(out_dir / "vqa_rad_train.jsonl", train_records, "train")
    write_jsonl(out_dir / "vqa_rad_val.jsonl", val_records, "val")
    write_jsonl(out_dir / "vqa_rad_test.jsonl", test_records, "test")

    stats = {
        "total_records": len(records),
        "unique_images": len({r["image"] for r in records}),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
        "closed_train": sum(1 for r in train_records if r["answer_type"] == "closed"),
        "open_train": sum(1 for r in train_records if r["answer_type"] == "open"),
    }
    stats_path = out_dir / "vqa_rad_split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Preprocessing complete.")
    print(json.dumps(stats, indent=2))
    print(f"Wrote: {out_dir / 'vqa_rad_train.jsonl'}")
    print(f"Wrote: {out_dir / 'vqa_rad_val.jsonl'}")
    print(f"Wrote: {out_dir / 'vqa_rad_test.jsonl'}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
