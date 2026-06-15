#!/usr/bin/env python3
"""Extract embedded images from VQA-RAD parquet files to disk."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def extract_split(parquet_path: Path, image_dir: Path, split_name: str) -> list[dict]:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    records: list[dict] = []

    for idx, row in enumerate(rows):
        image_info = row["image"]
        image_bytes = image_info.get("bytes")
        src_path = image_info.get("path") or f"{split_name}_{idx}.jpg"
        suffix = Path(src_path).suffix or ".jpg"
        image_name = f"{split_name}_{idx:04d}{suffix}"
        out_path = image_dir / image_name

        if image_bytes:
            Image.open(BytesIO(image_bytes)).convert("RGB").save(out_path)
        else:
            raise ValueError(f"Missing image bytes at {split_name}[{idx}]")

        records.append(
            {
                "image": str(out_path.relative_to(image_dir.parent.parent)),
                "question": row["question"],
                "answer": row["answer"],
                "split": split_name,
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/root/autodl-tmp/VQA/data/vqa-rad-raw"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/autodl-tmp/VQA/data"),
    )
    args = parser.parse_args()

    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    splits = {
        "train": args.raw_dir / "data/train-00000-of-00001-eb8844602202be60.parquet",
        "test": args.raw_dir / "data/test-00000-of-00001-e5bc3d208bb4deeb.parquet",
    }

    for split_name, parquet_path in splits.items():
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        print(f"Extracting {split_name}: {parquet_path.name}")
        records = extract_split(parquet_path, image_dir, split_name)
        all_records.extend(records)
        print(f"  -> {len(records)} samples")

    jsonl_path = args.out_dir / "vqa_rad_all.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_records)} records -> {jsonl_path}")
    print(f"Images -> {image_dir} ({len(list(image_dir.glob('*')))} files)")


if __name__ == "__main__":
    main()
