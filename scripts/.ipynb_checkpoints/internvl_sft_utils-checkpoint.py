"""Shared utilities for InternVL2 VQA-RAD SFT training and evaluation."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_infer import load_image_tensor

IGNORE_TOKEN_ID = -100
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMAGE_PLACEHOLDER = "<image>"

# InternLM2 (InternVL2-8B) attention/FFN module names for PEFT
INTERNLM2_LORA_TARGETS = [
    "attention.wqkv",
    "attention.wo",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
]


def ensure_model_on_path(model_path: Path) -> None:
    model_root = str(model_path.resolve())
    if model_root not in sys.path:
        sys.path.insert(0, model_root)


def get_conv_template(name: str):
    from conversation import get_conv_template as _get_conv_template

    return _get_conv_template(name)


@dataclass
class TrainConfig:
    project_root: Path
    model_path: Path
    train_file: Path
    val_file: Path
    output_dir: Path
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_length: int = 512
    max_num_patches: int = 6
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    train_projector: bool = True
    logging_dir: Path | None = None
    save_steps: int = 200
    eval_steps: int = 200
    bf16: bool = True
    seed: int = 42


def load_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def preprocess_internlm_conversation(
    template_name: str,
    conversations: list[dict],
    tokenizer: AutoTokenizer,
    num_image_tokens: int,
) -> dict[str, torch.Tensor]:
    """Tokenize one multi-modal sample and mask prompt tokens in labels."""
    conv = get_conv_template(template_name)
    role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conv = copy.deepcopy(conv)
    conv.messages = []
    for turn in conversations:
        role = role_map[turn["from"]]
        conv.append_message(role, turn["value"].strip())
    prompt = conv.get_prompt()

    image_token_str = (
        f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_tokens}{IMG_END_TOKEN}"
    )
    if IMAGE_PLACEHOLDER not in prompt:
        raise ValueError("Conversation must contain '<image>' placeholder.")
    prompt = prompt.replace(IMAGE_PLACEHOLDER, image_token_str, 1)

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]
    targets = input_ids.clone()

    total_len = int(targets.ne(tokenizer.pad_token_id).sum())
    cur_len = 1
    targets[:cur_len] = IGNORE_TOKEN_ID

    parts = prompt.split(conv.roles[1])
    info = parts[0] + conv.roles[1]
    temp_len = len(tokenizer(info, add_special_tokens=False).input_ids)
    targets[cur_len : cur_len + temp_len] = IGNORE_TOKEN_ID
    cur_len += temp_len

    for index in range(1, len(parts) - 1):
        info = parts[index]
        part1, part2 = info.split(conv.roles[0])
        temp_len = len(tokenizer(part1, add_special_tokens=False).input_ids)
        cur_len += temp_len
        part = conv.roles[0] + part2 + conv.roles[1]
        temp_len = len(tokenizer(part, add_special_tokens=False).input_ids)
        targets[cur_len : cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len += temp_len

    last_info = parts[-1]
    temp_len = len(tokenizer(last_info, add_special_tokens=False).input_ids)
    cur_len += temp_len
    targets[cur_len:] = IGNORE_TOKEN_ID

    if cur_len < tokenizer.model_max_length and cur_len != total_len:
        targets[:] = IGNORE_TOKEN_ID

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_mask,
    }


class InternVLSFTDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        project_root: Path,
        tokenizer: AutoTokenizer,
        template_name: str,
        num_image_token: int,
        max_num_patches: int = 6,
    ) -> None:
        self.records = records
        self.project_root = project_root
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.max_num_patches = max_num_patches

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        image_path = self.project_root / item["image"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        pixel_values = load_image_tensor(
            image_path, max_num=self.max_num_patches
        )
        num_patches = pixel_values.shape[0]

        encoded = preprocess_internlm_conversation(
            template_name=self.template_name,
            conversations=item["conversations"],
            tokenizer=self.tokenizer,
            num_image_tokens=self.num_image_token * num_patches,
        )

        return {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": pixel_values,
            "image_flags": torch.ones(num_patches, dtype=torch.long),
            "id": item.get("id", str(index)),
        }


def collate_internvl_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [sample["input_ids"] for sample in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        [sample["labels"] for sample in batch],
        batch_first=True,
        padding_value=IGNORE_TOKEN_ID,
    )
    attention_mask = input_ids.ne(pad_token_id)
    pixel_values = torch.cat([sample["pixel_values"] for sample in batch], dim=0)
    image_flags = torch.cat([sample["image_flags"] for sample in batch], dim=0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_flags": image_flags,
    }


class InternVLTrainer(Trainer):
    def __init__(self, *args, img_context_token_id: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_context_token_id = img_context_token_id

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model.img_context_token_id = self.img_context_token_id
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def load_tokenizer(model_path: Path) -> AutoTokenizer:
    ensure_model_on_path(model_path)
    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )


def load_base_model(model_path: Path, use_flash_attn: bool = False) -> AutoModel:
    ensure_model_on_path(model_path)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401

            model_kwargs["use_flash_attn"] = True
        except ImportError:
            pass
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    return model


def apply_lora(
    model: AutoModel,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    train_projector: bool = False,
) -> AutoModel:
    for param in model.vision_model.parameters():
        param.requires_grad = False

    if not train_projector:
        for param in model.mlp1.parameters():
            param.requires_grad = False

    model.config.use_cache = False
    if hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=INTERNLM2_LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.enable_input_require_grads()
    if hasattr(model.language_model, "gradient_checkpointing_enable"):
        model.language_model.gradient_checkpointing_enable()
    model.language_model.print_trainable_parameters()
    return model


def load_model_with_lora(
    model_path: Path,
    lora_path: Path,
    use_flash_attn: bool = False,
) -> tuple[AutoModel, AutoTokenizer]:
    tokenizer = load_tokenizer(model_path)
    model = load_base_model(model_path, use_flash_attn=use_flash_attn)
    model.language_model = PeftModel.from_pretrained(
        model.language_model, lora_path, is_trainable=False
    )
    model.eval().cuda()
    return model, tokenizer


def build_training_arguments(cfg: TrainConfig) -> TrainingArguments:
    logging_dir = cfg.logging_dir or (cfg.project_root / "outputs/tb_logs")
    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=cfg.bf16 and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        logging_dir=str(logging_dir),
        seed=cfg.seed,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )
