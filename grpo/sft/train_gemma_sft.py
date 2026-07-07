"""
train_gemma_sft.py
------------------
SFTs Gemma 4 E4B on the multi-level budget-compression dataset using the
fixed CompletionOnlyCollator pattern from retrain_l4.py.

Why a separate script:
  The stored `prompt` column in the Arrow dataset was built with Qwen3's
  chat template (<|im_start|> tokens).  Gemma uses a different template
  (<start_of_turn>) so we rebuild prompts from the raw `question` and `level`
  fields, using Gemma's tokenizer.apply_chat_template.

Key fixes carried over from retrain_l4.py:
  - CompletionOnlyCollator uses precomputed prompt_length (no template search)
  - remove_unused_columns=False in SFTConfig (prevents KeyError: prompt_length)

Usage:
  python3 train_gemma_sft.py                # all levels 1-5 in sequence
  python3 train_gemma_sft.py --level 4      # single level
  python3 train_gemma_sft.py --level 4 --epochs 2 --bs 4

UPDATE BASE_MODEL below to the exact HuggingFace model ID you want
(e.g. "google/gemma-3-4b-it" or "google/gemma-4e-4b-it").
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "google/gemma-3-4b-it"   # UPDATE: exact HF model ID
DATA_ROOT    = "./initial_data/prepared_train"
OUTPUT_ROOT  = "./checkpoints_gemma"

EPOCHS      = 3
LR          = 2e-4
BS          = 8           # Gemma 3 4B; increase if VRAM allows
GRAD_ACCUM  = 8           # effective batch = 64
MAX_SEQ     = 1024
LORA_R      = 16
LORA_ALPHA  = 32
LORA_DROP   = 0.05
SAVE_STEPS  = 200
LOG_STEPS   = 10

LEVEL_LABELS = {
    1: "Level 1 (Verbose explanation)",
    2: "Level 2 (Structured shorthand)",
    3: "Level 3 (Variable chain)",
    4: "Level 4 (Ultra-compact)",
    5: "Level 5 (Pure expression)",
}
# ─────────────────────────────────────────────────────────────────────────────


class CompletionOnlyCollator:
    """
    Masks prompt tokens from the loss using precomputed prompt_length.

    Avoids the template-search failure mode where tokenizer.encode() of the
    standalone response template produces different IDs than in-context
    tokenization — causing mask_until=0 (no masking) and the model learning
    prompt tokens as targets.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id    = tokenizer.pad_token_id

    def __call__(self, features):
        clean = [
            {"input_ids": f["input_ids"],
             **({"attention_mask": f["attention_mask"]} if "attention_mask" in f else {})}
            for f in features
        ]
        batch  = self.tokenizer.pad(clean, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        for i, f in enumerate(features):
            prompt_len = f["prompt_length"]
            ids        = batch["input_ids"][i].tolist()
            # Account for left-padding added by tokenizer.pad()
            pad_count  = sum(1 for t in ids if t == self.pad_id)
            labels[i, : pad_count + prompt_len] = -100

        batch["labels"] = labels
        return batch


def build_gemma_examples(dataset, tokenizer, level_label: str):
    """
    Rebuild every example for Gemma's chat template.

    The stored `prompt`/`text` columns use Qwen3 tokens.  We reconstruct:
      - text         : apply_chat_template over full conversation
      - prompt_length: number of tokens in the prompt-only prefix
    """
    qwen_eos = "<|im_end|>"

    def _reformat(example):
        instruction = f"Solve this using {level_label}.\nProblem: {example['question']}"

        # Strip Qwen3-specific end token from completion content
        completion_content = example["completion"]
        if completion_content.endswith(qwen_eos):
            completion_content = completion_content[: -len(qwen_eos)]

        messages = [
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": completion_content},
        ]

        # Full conversation text (Gemma adds its own EOS)
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Prompt-only prefix for length computation
        prompt_text = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids    = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_ids)

        return {
            "text":         full_text,
            "prompt_length": prompt_length,
        }

    return dataset.map(_reformat, desc="reformatting for Gemma", num_proc=4)


def load_tokenizer(base_model: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
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
        use_cache=False,
    )


def train_level(level: int, tokenizer, model_name: str, args):
    label    = LEVEL_LABELS[level]
    data_dir = Path(args.data_root) / f"level_{level}"
    out_dir  = Path(args.output_root) / f"level_{level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[gemma_sft] Level {level} — {label}")
    print(f"[gemma_sft] data : {data_dir}")
    print(f"[gemma_sft] out  : {out_dir}")

    ds       = load_from_disk(str(data_dir))
    train_ds = ds["train"]
    val_ds   = ds["validation"]
    print(f"[gemma_sft] train={len(train_ds)}  val={len(val_ds)}")

    print("[gemma_sft] Reformatting examples for Gemma chat template ...")
    train_ds = build_gemma_examples(train_ds, tokenizer, label)
    val_ds   = build_gemma_examples(val_ds,   tokenizer, label)

    # Sanity check: print first completion token (should be start of <think>)
    for i in range(3):
        ex       = train_ds[i]
        p_len    = ex["prompt_length"]
        full_ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        first_id = full_ids[p_len] if p_len < len(full_ids) else -1
        first_tok = tokenizer.decode([first_id])
        print(f"  ex[{i}]: prompt_length={p_len}  first_completion_tok={repr(first_tok)}")

    print(f"[gemma_sft] Loading model {model_name}")
    model = load_model(model_name)

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
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
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
        max_seq_length=args.max_seq,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,   # keeps prompt_length column in features
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=42,
        run_name=f"gemma-4b-sft-l{level}",
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

    print(f"[gemma_sft] Training level {level} ...")
    trainer.train()

    final_dir = out_dir / "final_adapter"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[gemma_sft] Done. Adapter → {final_dir}")

    # Free model memory before next level
    del model
    del trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="SFT Gemma on budget-compression levels")
    parser.add_argument("--model",        default=BASE_MODEL)
    parser.add_argument("--level",        type=int, default=None,
                        help="Single level to train (1-5). Omit for all.")
    parser.add_argument("--data-root",    default=DATA_ROOT)
    parser.add_argument("--output-root",  default=OUTPUT_ROOT)
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--bs",           type=int,   default=BS)
    parser.add_argument("--grad-accum",   type=int,   default=GRAD_ACCUM)
    parser.add_argument("--max-seq",      type=int,   default=MAX_SEQ)
    args = parser.parse_args()

    levels = [args.level] if args.level else sorted(LEVEL_LABELS.keys())

    print(f"[gemma_sft] Base model : {args.model}")
    print(f"[gemma_sft] Levels     : {levels}")
    print(f"[gemma_sft] Epochs     : {args.epochs}")
    print(f"[gemma_sft] BS         : {args.bs}  grad_accum={args.grad_accum}"
          f"  (eff={args.bs * args.grad_accum})")
    print(f"[gemma_sft] LR         : {args.lr}")

    # Load tokenizer once; model is reloaded per level to free VRAM
    print("[gemma_sft] Loading tokenizer ...")
    tokenizer = load_tokenizer(args.model)

    for level in levels:
        train_level(level, tokenizer, args.model, args)

    print("\n[gemma_sft] All levels complete.")


if __name__ == "__main__":
    main()
