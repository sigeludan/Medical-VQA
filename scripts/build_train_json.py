#!/usr/bin/env python3
"""Convert VQA-RAD JSONL splits into InternVL2 SFT JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_infer_v2 import build_prompt
from vqa_common import normalize_text

PROMPT_VERSION = "v2"
IMAGE_PLACEHOLDER = "<image>"


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_internvl_record(row: dict, index: int, split_name: str) -> dict:
    question = normalize_text(row["question"])
    answer = normalize_text(row["answer"])
    prompt = build_prompt(question)
    human_value = f"{IMAGE_PLACEHOLDER}\n{prompt}"

    image_path = row["image"]
    record_id = Path(image_path).stem or f"{split_name}_{index:04d}"

    return {
        "id": record_id,
        "image": image_path,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": answer},
        ],
        "question": question,
        "answer": answer,
        "answer_type": row.get("answer_type", "closed"),
        "split": split_name,
        "prompt_version": PROMPT_VERSION,
    }


def convert_split(jsonl_path: Path, split_name: str) -> list[dict]:
    rows = load_jsonl(jsonl_path)
    return [to_internvl_record(row, idx, split_name) for idx, row in enumerate(rows)]


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build InternVL2 SFT JSON from VQA-RAD JSONL.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    train_jsonl = args.train_jsonl or data_dir / "vqa_rad_train.jsonl"
    val_jsonl = args.val_jsonl or data_dir / "vqa_rad_val.jsonl"
    out_dir = args.out_dir or data_dir

    train_records = convert_split(train_jsonl, "train")
    val_records = convert_split(val_jsonl, "val")

    train_out = out_dir / "train_internvl.json"
    val_out = out_dir / "val_internvl.json"
    write_json(train_out, train_records)
    write_json(val_out, val_records)

    stats = {
        "prompt_version": PROMPT_VERSION,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_closed": sum(1 for r in train_records if r["answer_type"] == "closed"),
        "train_open": sum(1 for r in train_records if r["answer_type"] == "open"),
        "train_file": str(train_out),
        "val_file": str(val_out),
        "sample": train_records[0] if train_records else None,
    }
    stats_path = out_dir / "train_internvl_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("InternVL SFT JSON ready.")
    print(json.dumps({k: v for k, v in stats.items() if k != "sample"}, indent=2))
    print(f"Wrote: {train_out}")
    print(f"Wrote: {val_out}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
