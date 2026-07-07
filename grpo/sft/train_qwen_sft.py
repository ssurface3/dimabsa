"""
retrain_l4.py
-------------
Re-trains SFT for Level 4 (Ultra-compact) only, with the completion-only
loss masking bug fixed.

Bug in original CompletionOnlyCollator: searched for response_template token
IDs via pattern match. When tokenization of the standalone template differed
from in-context tokenization, mask_until stayed 0 → labels[:0]=-100 masked
nothing → model trained on prompt tokens too → tool_call format corruption.

Fix: precompute exact prompt token lengths at dataset load time and use them
directly in the collator. No pattern search, can't fail silently.

Usage:
    python3 retrain_l4.py

Edit the CONFIG block below if paths differ on your machine.
"""

from __future__ import annotations

import os
import torch
from dataclasses import dataclass
from pathlib import Path

from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
DATA_DIR    = "./initial_data/prepared_train/level_4"
OUTPUT_DIR  = "./checkpoints_new/level_4_fixed"

EPOCHS      = 3          # 3 instead of 2 — extra epoch helps override tool-call prior
LR          = 2e-4
BS          = 16
GRAD_ACCUM  = 4          # effective batch size = 64
MAX_SEQ     = 1024
LORA_R      = 16
LORA_ALPHA  = 32
LORA_DROP   = 0.05
SAVE_STEPS  = 200
LOG_STEPS   = 10
# ─────────────────────────────────────────────────────────────────────────────


class CompletionOnlyCollator:
    """
    Masks prompt tokens from the loss using precomputed prompt_length.

    Avoids the template-search failure mode where tokenizer.encode() of the
    standalone response template produces different IDs than in-context
    tokenization — causing mask_until=0 (no masking) and the model learning
    prompt tokens as if they were targets.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        clean = [
            {"input_ids": f["input_ids"],
             **({"attention_mask": f["attention_mask"]} if "attention_mask" in f else {})}
            for f in features
        ]
        batch = self.tokenizer.pad(clean, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        for i, f in enumerate(features):
            prompt_len = f["prompt_length"]
            ids = batch["input_ids"][i].tolist()
            # Account for left-padding added by tokenizer.pad()
            pad_count = sum(1 for t in ids if t == self.pad_id)
            # Mask pad tokens + all prompt tokens
            labels[i, : pad_count + prompt_len] = -100

        batch["labels"] = labels
        return batch


def add_prompt_lengths(dataset, tokenizer):
    """Precompute prompt token lengths so the collator doesn't need template search."""
    def _compute(example):
        ids = tokenizer(
            example["prompt"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        example["prompt_length"] = len(ids)
        return example
    return dataset.map(_compute, desc="computing prompt lengths", num_proc=4)


def load_tokenizer(base_model: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(base_model: str) -> AutoModelForCausalLM:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[retrain_l4] Loading dataset from {DATA_DIR}")
    ds = load_from_disk(DATA_DIR)
    train_ds = ds["train"]
    val_ds   = ds["validation"]
    print(f"[retrain_l4] train={len(train_ds)}  val={len(val_ds)}")

    tokenizer = load_tokenizer(BASE_MODEL)

    print("[retrain_l4] Precomputing prompt lengths...")
    train_ds = add_prompt_lengths(train_ds, tokenizer)
    val_ds   = add_prompt_lengths(val_ds,   tokenizer)

    # Sanity check: print a few prompt lengths + first completion token
    for i in range(3):
        ex = train_ds[i]
        p_len = ex["prompt_length"]
        full_ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        comp_start_id = full_ids[p_len] if p_len < len(full_ids) else -1
        comp_start_tok = tokenizer.decode([comp_start_id])
        print(f"  ex[{i}]: prompt_length={p_len}  first_completion_token={repr(comp_start_tok)}")

    print(f"[retrain_l4] Loading model {BASE_MODEL}")
    model = load_model(BASE_MODEL)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROP,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    collator = CompletionOnlyCollator(tokenizer)

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BS,
        per_device_eval_batch_size=BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=LOG_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=False,
        max_seq_length=MAX_SEQ,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=42,
        run_name="qwen3-4b-l4-fixed",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        peft_config=lora_config,
        data_collator=collator,
    )

    print(f"[retrain_l4] Starting training → {out_dir}")
    trainer.train()

    final_dir = out_dir / "final_adapter"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[retrain_l4] Done. Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
