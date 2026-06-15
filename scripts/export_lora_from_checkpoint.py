#!/usr/bin/env python3
"""Export a PEFT LoRA adapter from a HuggingFace Trainer full-model checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from internvl_sft_utils import apply_lora, ensure_model_on_path, load_base_model


def load_checkpoint_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    shard_paths = sorted(checkpoint.glob("model-*.safetensors"))
    if not shard_paths:
        single = checkpoint / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"No safetensors weights found in {checkpoint}")
        return load_file(str(single))
    for shard in shard_paths:
        state.update(load_file(str(shard)))
    return state


def extract_peft_state_dict(full_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    peft_state: dict[str, torch.Tensor] = {}
    for key, value in full_state.items():
        if "lora_" not in key:
            continue
        if not key.startswith("language_model."):
            continue
        peft_key = key[len("language_model.") :]
        peft_state[peft_key] = value
    if not peft_state:
        raise RuntimeError("No LoRA weights found in checkpoint state dict.")
    return peft_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LoRA adapter from Trainer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    project_root = Path(__file__).resolve().parents[1]
    base_model = (args.base_model or project_root / "models/OpenGVLab/InternVL2-8B").resolve()
    output_dir = args.output_dir.resolve()
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ensure_model_on_path(base_model)
    print(f"Loading base model: {base_model}")
    model = load_base_model(base_model, use_flash_attn=False)
    model = apply_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        train_projector=False,
    )

    print(f"Loading checkpoint weights: {checkpoint}")
    full_state = load_checkpoint_state_dict(checkpoint)
    peft_state = extract_peft_state_dict(full_state)
    print(f"Found {len(peft_state)} LoRA tensors")

    incompatible = set_peft_model_state_dict(model.language_model, peft_state)
    if incompatible:
        print(f"Warning: {len(incompatible)} unexpected keys while loading LoRA state")

    local_base = str(base_model)
    peft_model = model.language_model
    if hasattr(peft_model, "peft_config"):
        for key in peft_model.peft_config:
            peft_model.peft_config[key].base_model_name_or_path = local_base

    print(f"Saving adapter -> {adapter_dir}")
    peft_model.save_pretrained(adapter_dir, safe_serialization=True)

    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if adapter_cfg_path.exists():
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        adapter_cfg["base_model_name_or_path"] = local_base
        adapter_cfg_path.write_text(
            json.dumps(adapter_cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    state_path = checkpoint / "trainer_state.json"
    meta = {
        "base_model": local_base,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "source_checkpoint": str(checkpoint),
        "adapter_dir": str(adapter_dir),
        "exported_from_trainer_checkpoint": True,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "best_eval_loss": state.get("best_metric"),
                "best_checkpoint": state.get("best_model_checkpoint"),
                "global_step": state.get("global_step"),
                "epoch": state.get("epoch"),
            }
        )
    (output_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    adapter_size = sum(p.stat().st_size for p in adapter_dir.iterdir() if p.is_file())
    print(f"Adapter size: {adapter_size / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
