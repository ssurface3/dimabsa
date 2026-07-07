"""
train_gemma_full.py
-------------------
Combined SFT → GRPO pipeline for Gemma on budget-compression levels.

Phase 1 (SFT)
  Fine-tunes Gemma on the per-level Arrow datasets (initial_data/prepared_train/).
  Prompts are rebuilt from raw question text using Gemma's chat template
  (stored prompts use Qwen3 tokens and cannot be reused).
  CompletionOnlyCollator uses precomputed prompt_length — no template search,
  can't fail silently. remove_unused_columns=False prevents KeyError.

Phase 2 (GRPO)
  Loads the SFT adapter, merges it into the base model in-memory, adds a fresh
  GRPO LoRA, then trains with GRPOTrainer on the balanced-1k GRPO dataset.
  Rewards: correctness + format + cleanliness, with length as GDPO passthrough.
  Reward functions inlined — no imports from logic_train/.

Usage:
  python3 train_gemma_full.py                         # both phases, all levels 1-5
  python3 train_gemma_full.py --phase sft             # SFT only
  python3 train_gemma_full.py --phase grpo            # GRPO only (SFT ckpt must exist)
  python3 train_gemma_full.py --level 4               # one level
  python3 train_gemma_full.py --level 4 --phase grpo --sft-dir ./checkpoints_gemma_sft
  python3 train_gemma_full.py --rewards correctness,format,length,gdpo

UPDATE BASE_MODEL below to the exact HuggingFace model ID
(e.g. "google/gemma-3-4b-it" or "google/gemma-4e-4b-it").
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer


# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "google/gemma-3-4b-it"   # UPDATE to exact HF model ID
SFT_DATA_ROOT    = "./initial_data/prepared_train"
GRPO_DATASET     = "./gsm8k_grpo_balanced_1k.json"
SFT_OUTPUT_ROOT  = "./checkpoints_gemma_sft"
GRPO_OUTPUT_ROOT = "./checkpoints_gemma_grpo"

# SFT hyperparams
SFT_EPOCHS      = 3
SFT_LR          = 2e-4
SFT_BS          = 8
SFT_GRAD_ACCUM  = 8    # effective batch = 64
SFT_MAX_SEQ     = 1024

# GRPO hyperparams
GRPO_EPOCHS     = 1
GRPO_LR         = 1e-5
GRPO_BS         = 4
GRPO_ACCUM      = 2
GRPO_NUM_GEN    = 8
GRPO_MAX_COMP   = 256

# Shared LoRA
LORA_R     = 16
LORA_ALPHA = 32
LORA_DROP  = 0.05

LEVEL_LABELS = {
    1: "Level 1 (Verbose explanation)",
    2: "Level 2 (Structured shorthand)",
    3: "Level 3 (Variable chain)",
    4: "Level 4 (Ultra-compact)",
    5: "Level 5 (Pure expression)",
}
LEVEL_KEYS = {lvl: f"level {lvl}" for lvl in LEVEL_LABELS}

# Gemma 4 ships without a chat_template in the tokenizer config.
# This is the standard Gemma format used by Gemma 2/3/4.
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] | trim }}<end_of_turn>\n<start_of_turn>model\n"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] | trim }}<end_of_turn>\n"
    "{% endif %}"
    "{% endfor %}"
)
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SFT utilities
# ══════════════════════════════════════════════════════════════════════════════

class CompletionOnlyCollator:
    """
    Masks prompt tokens using precomputed prompt_length — no template search.
    Avoids silent mask_until=0 bug where tokenizer IDs differ in/out of context.
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
            ids       = batch["input_ids"][i].tolist()
            pad_count = sum(1 for t in ids if t == self.pad_id)
            labels[i, : pad_count + f["prompt_length"]] = -100
        batch["labels"] = labels
        return batch


def build_sft_examples(dataset, tokenizer, level_label: str):
    """Rebuild text + prompt_length from Gemma's chat template."""
    qwen_eos = "<|im_end|>"

    def _reformat(example):
        instruction = f"Solve this using {level_label}.\nProblem: {example['question']}"
        completion_content = example["completion"]
        if completion_content.endswith(qwen_eos):
            completion_content = completion_content[: -len(qwen_eos)]

        messages = [
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": completion_content},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return {"text": full_text, "prompt_length": len(prompt_ids)}

    return dataset.map(_reformat, desc="reformatting for Gemma", num_proc=4)


# ══════════════════════════════════════════════════════════════════════════════
# GRPO dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_grpo_dataset(path: str, level: int) -> Dataset:
    """
    Load the balanced-1k GRPO JSON, filter to one level, and return a Dataset
    whose 'prompt' column is a list of message dicts (TRL applies chat template).
    """
    with open(path) as f:
        rows = json.load(f)

    level_key   = LEVEL_KEYS[level]
    level_label = LEVEL_LABELS[level]
    level_rows  = [r for r in rows if r["level"] == level_key]
    print(f"[grpo] dataset: {len(level_rows)} rows for {level_key}")

    records = []
    for row in level_rows:
        # prompt is a list of message dicts (json.load parses it natively)
        content  = row["prompt"][0]["content"]
        question = content.split("Problem: ", 1)[1] if "Problem: " in content else content

        records.append({
            # TRL GRPOTrainer: list-of-dicts → apply_chat_template automatically
            "prompt":       [{"role": "user",
                              "content": f"Solve this using {level_label}.\nProblem: {question}"}],
            "ground_truth": str(row["ground_truth"]),
            "complexity":   str(row["complexity"]),
            "level":        row["level"],
        })

    return Dataset.from_list(records)


# ══════════════════════════════════════════════════════════════════════════════
# Reward functions (inlined — no import from logic_train/)
# ══════════════════════════════════════════════════════════════════════════════

def correctness_reward(prompts, completions, ground_truth, complexity, **kwargs):
    rewards = []
    for completion, gt, cplx in zip(completions, ground_truth, complexity):
        content   = completion[0]["content"]
        cplx      = int(cplx)
        match     = re.search(r"###?#?\s*([\d\.,]+)", content)
        predicted = match.group(1).strip().replace(",", "") if match else ""
        rewards.append(cplx if predicted == gt.strip() else -cplx)
    return rewards


def goldilocks_reward(completions, level, **kwargs):
    """Double-hinge penalty: too short → -0.05/char under, too long → -0.01/char over."""
    ranges = {
        "level 1": (120, 240),
        "level 2": (50,  110),
        "level 3": (40,  100),
        "level 4": (20,   60),
        "level 5": (10,   30),
    }
    rewards = []
    for completion, lvl in zip(completions, level):
        content = completion[0]["content"]
        m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        reasoning = m.group(1) if m else content
        n = len(reasoning)
        lo, hi = ranges.get(str(lvl).lower().strip(), (40, 100))
        if n < lo:
            rewards.append((n - lo) * 0.05)   # negative
        elif n > hi:
            rewards.append((hi - n) * 0.01)   # negative
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        content    = completion[0]["content"]
        has_think  = bool(re.search(r"<think>.*?</think>", content, re.DOTALL)) or \
                     bool(re.search(r"</think>", content))
        has_answer = bool(re.search(r"###?#?\s*[\d\.,]+", content))
        if has_think and has_answer:
            rewards.append(1.0)
        elif has_answer and not has_think:
            rewards.append(-3.0)
        elif has_think and not has_answer:
            rewards.append(-1.0)
        else:
            rewards.append(-5.0)
    return rewards


_ALLOWED_NON_ASCII = frozenset([
    '→','×','÷','−','Δ','≤','≥','∈','≈','≠','Σ','⇒','∩','…',
    '€','£','¢','°','²','³','·','‑','–','’','⌈','⌉','∀','∃','¼','½','¾',
    '₀','₁','₂','₃','₄','₅','₆','₇','₈','₉',
    '⁰','¹','⁴','⁵','⁶','⁷','⁸','⁹',
    'α','β','γ','δ','θ','λ','μ','π','σ','φ','ω','Α','Β','Γ','Λ','Π','Φ','Ω',
])

def cleanliness_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        garbage = sum(1 for ch in content if ord(ch) > 127 and ch not in _ALLOWED_NON_ASCII)
        rewards.append(-min(garbage * 0.02, 3.0))
    return rewards


_LENGTH_TARGETS = {
    "level 1": (60,  220),
    "level 2": (30,  130),
    "level 3": (25,  120),
    "level 4": (12,   80),
    "level 5": ( 6,   50),
}

def length_reward(completions, level, ground_truth, **kwargs):
    rewards = []
    for completion, lvl, gt in zip(completions, level, ground_truth):
        content    = completion[0]["content"]
        match_ans  = re.search(r"###?#?\s*([\d\.,]+)", content)
        predicted  = match_ans.group(1).replace(",", "") if match_ans else ""
        correct    = (predicted == str(gt).strip())
        m          = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        think_chars = len(m.group(1)) if m else 0
        min_t, max_t = _LENGTH_TARGETS.get(str(lvl).lower().strip(), (15, 100))
        if think_chars < 5:
            penalty = -3.0
        elif think_chars < min_t and correct:
            penalty = -1.5 * (min_t - think_chars) / min_t
        elif think_chars > max_t and correct:
            penalty = -0.01 * (think_chars - max_t)
        else:
            penalty = 0.0
        rewards.append(penalty)
    return rewards


_LEVEL_COT_DEFAULTS = {
    "level 1": 532, "level 2": 140, "level 3": 90, "level 4": 41, "level 5": 16,
}

def sft_length_reward(completions, cot_chars=None, level=None, **kwargs):
    """
    Exponential penalty when think content exceeds the per-row SFT target (cot_chars).
    Zero when at or under target. Grows fast above it:
      +100 chars over → -1.72,  +200 → -6.39,  +300 → -19.1
    Falls back to per-level medians from SFT data when cot_chars is not forwarded.
    """
    SCALE = 100
    if cot_chars is None:
        levels    = level if level is not None else ["level 3"] * len(completions)
        cot_chars = [_LEVEL_COT_DEFAULTS.get(str(lvl).lower().strip(), 90) for lvl in levels]

    rewards = []
    for completion, target in zip(completions, cot_chars):
        content     = completion[0]["content"]
        m           = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        think_chars = len(m.group(1)) if m else len(content)
        excess      = max(0, think_chars - int(target))
        rewards.append(-(math.exp(excess / SCALE) - 1.0) if excess > 0 else 0.0)
    return rewards


def make_gdpo_reward(reward_funcs, num_gen=8, passthrough_funcs=None):
    """
    GDPO: normalize each reward independently per generation group before summing.
    passthrough_funcs bypass normalization — use for length (survives group collapse).
    """
    _passthrough = passthrough_funcs or []

    def _combined(prompts, completions, **kwargs):
        all_rewards = [fn(prompts=prompts, completions=completions, **kwargs)
                       for fn in reward_funcs]
        n        = len(all_rewards[0])
        n_groups = max(1, n // num_gen)
        combined = [0.0] * n

        for rewards in all_rewards:
            for g in range(n_groups):
                start = g * num_gen
                end   = min(start + num_gen, n)
                grp   = rewards[start:end]
                mean_g = sum(grp) / len(grp)
                std_g  = (sum((x - mean_g) ** 2 for x in grp) / len(grp)) ** 0.5 + 1e-8
                for k, idx in enumerate(range(start, end)):
                    combined[idx] += (grp[k] - mean_g) / std_g

        for fn in _passthrough:
            r = fn(prompts=prompts, completions=completions, **kwargs)
            for i, val in enumerate(r):
                combined[i] += val

        return combined

    return _combined


REWARD_REGISTRY = {
    "correctness": correctness_reward,
    "goldilocks":  goldilocks_reward,
    "format":      format_reward,
    "cleanliness": cleanliness_reward,
    "length":      length_reward,
    "sft_length":  sft_length_reward,
}
PASSTHROUGH_NAMES = {"length"}


def build_reward_func(reward_str: str, num_gen: int):
    """Parse reward string → single callable for GRPOTrainer."""
    names     = [r.strip() for r in reward_str.split(",")]
    use_gdpo  = "gdpo" in names
    names     = [n for n in names if n != "gdpo"]

    for n in names:
        if n not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward '{n}'. Available: {list(REWARD_REGISTRY)}")

    main_funcs = [REWARD_REGISTRY[n] for n in names if n not in PASSTHROUGH_NAMES]
    pass_funcs = [REWARD_REGISTRY[n] for n in names if n in PASSTHROUGH_NAMES]

    if use_gdpo:
        return [make_gdpo_reward(main_funcs, num_gen=num_gen, passthrough_funcs=pass_funcs)], [1.0]
    else:
        funcs   = main_funcs + pass_funcs
        weights = [1.0] * len(funcs)
        return funcs, weights


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if not tok.chat_template:
        tok.chat_template = GEMMA_CHAT_TEMPLATE
    return tok


def load_base_model(model_name: str) -> AutoModelForCausalLM:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": local_rank},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model


def make_lora_config(dropout=LORA_DROP) -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# SFT phase
# ══════════════════════════════════════════════════════════════════════════════

def train_sft_level(level: int, tokenizer, args) -> Path:
    label   = LEVEL_LABELS[level]
    out_dir = Path(args.sft_dir) / f"level_{level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[SFT] Level {level} — {label}")
    print(f"[SFT] output: {out_dir}")

    data_dir = Path(args.sft_data) / f"level_{level}"
    ds       = load_from_disk(str(data_dir))
    train_ds = build_sft_examples(ds["train"],      tokenizer, label)
    val_ds   = build_sft_examples(ds["validation"], tokenizer, label)
    print(f"[SFT] train={len(train_ds)}  val={len(val_ds)}")

    # Sanity: first completion token should be start of <think>
    for i in range(3):
        ex       = train_ds[i]
        full_ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        p_len    = ex["prompt_length"]
        first_id = full_ids[p_len] if p_len < len(full_ids) else -1
        print(f"  ex[{i}]: prompt_len={p_len}  first_completion={repr(tokenizer.decode([first_id]))}")

    model = load_base_model(args.model)

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.sft_bs,
        per_device_eval_batch_size=args.sft_bs,
        gradient_accumulation_steps=args.sft_grad_accum,
        learning_rate=args.sft_lr,
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        fp16=False, bf16=True, tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=10,
        eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False,
        max_seq_length=args.sft_max_seq,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,   # keeps prompt_length column
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=42,
        run_name=f"gemma-sft-l{level}",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        peft_config=make_lora_config(),
        data_collator=CompletionOnlyCollator(tokenizer),
    )

    print(f"[SFT] Training level {level} ...")
    trainer.train()

    adapter_dir = out_dir / "final_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[SFT] Adapter saved → {adapter_dir}")

    del model, trainer
    torch.cuda.empty_cache()
    return adapter_dir


# ══════════════════════════════════════════════════════════════════════════════
# GRPO phase
# ══════════════════════════════════════════════════════════════════════════════

def train_grpo_level(level: int, tokenizer, sft_adapter: Path, args):
    label   = LEVEL_LABELS[level]
    out_dir = Path(args.grpo_dir) / f"level_{level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[GRPO] Level {level} — {label}")
    print(f"[GRPO] SFT adapter: {sft_adapter}")
    print(f"[GRPO] output:      {out_dir}")

    dataset = build_grpo_dataset(args.grpo_dataset, level)

    # Merge SFT adapter into base → clean slate for GRPO LoRA
    print(f"[GRPO] Loading base + merging SFT adapter ...")
    base      = load_base_model(args.model)
    sft_model = PeftModel.from_pretrained(base, str(sft_adapter))
    merged    = sft_model.merge_and_unload()
    merged.enable_input_require_grads()

    grpo_model = get_peft_model(merged, make_lora_config(dropout=0.0))
    grpo_model.print_trainable_parameters()

    reward_funcs, reward_weights = build_reward_func(args.rewards, args.grpo_num_gen)

    print(f"[GRPO] Rewards: {args.rewards}")

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=args.grpo_lr,
        per_device_train_batch_size=args.grpo_bs,
        gradient_accumulation_steps=args.grpo_accum,
        num_train_epochs=args.grpo_epochs,
        num_generations=args.grpo_num_gen,
        max_completion_length=args.grpo_max_comp,
        loss_type="dapo",
        beta=0,
        reward_weights=reward_weights,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        seed=42,
        run_name=f"gemma-grpo-l{level}",
    )

    # Resume from last checkpoint if interrupted
    last_ckpt = None
    if out_dir.is_dir():
        ckpts = sorted(
            [d for d in out_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if ckpts:
            last_ckpt = str(ckpts[-1])
            print(f"[GRPO] Resuming from {last_ckpt}")

    trainer = GRPOTrainer(
        model=grpo_model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(str(out_dir / "final_adapter"))
    print(f"[GRPO] Adapter saved → {out_dir / 'final_adapter'}")

    del merged, grpo_model, trainer
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Gemma SFT + GRPO pipeline")

    p.add_argument("--model",          default=BASE_MODEL,
                   help="HuggingFace model ID")
    p.add_argument("--phase",          default="both",
                   choices=["sft", "grpo", "both"])
    p.add_argument("--level",          type=int, default=None,
                   help="Single level to train (1-5). Omit for all.")

    # Paths
    p.add_argument("--sft-data",       default=SFT_DATA_ROOT)
    p.add_argument("--sft-dir",        default=SFT_OUTPUT_ROOT,
                   help="Where SFT adapters are saved / loaded from")
    p.add_argument("--grpo-dataset",   default=GRPO_DATASET)
    p.add_argument("--grpo-dir",       default=GRPO_OUTPUT_ROOT)

    # SFT hyperparams
    p.add_argument("--sft-epochs",     type=int,   default=SFT_EPOCHS)
    p.add_argument("--sft-lr",         type=float, default=SFT_LR)
    p.add_argument("--sft-bs",         type=int,   default=SFT_BS)
    p.add_argument("--sft-grad-accum", type=int,   default=SFT_GRAD_ACCUM)
    p.add_argument("--sft-max-seq",    type=int,   default=SFT_MAX_SEQ)

    # GRPO hyperparams
    p.add_argument("--grpo-epochs",    type=int,   default=GRPO_EPOCHS)
    p.add_argument("--grpo-lr",        type=float, default=GRPO_LR)
    p.add_argument("--grpo-bs",        type=int,   default=GRPO_BS)
    p.add_argument("--grpo-accum",     type=int,   default=GRPO_ACCUM)
    p.add_argument("--grpo-num-gen",   type=int,   default=GRPO_NUM_GEN)
    p.add_argument("--grpo-max-comp",  type=int,   default=GRPO_MAX_COMP)
    p.add_argument("--rewards",        default="correctness,format,sft_length,gdpo",
                   help="Comma-separated reward names + optional 'gdpo' wrapper")

    return p.parse_args()


def preflight_check(args, levels):
    """
    Verify that all required files and directories exist before any GPU work starts.
    Prints a full status table, then raises SystemExit if anything is missing.
    """
    errors = []

    def ok(label):
        print(f"  [OK]   {label}")

    def fail(label, reason=""):
        msg = f"  [FAIL] {label}" + (f"  — {reason}" if reason else "")
        print(msg)
        errors.append(label)

    print(f"\n{'='*60}")
    print("PREFLIGHT CHECK")
    print(f"{'='*60}")

    # ── 1. reward names valid ──────────────────────────────────
    print("\n[rewards]")
    reward_names = [r.strip() for r in args.rewards.split(",") if r.strip() != "gdpo"]
    for name in reward_names:
        if name in REWARD_REGISTRY:
            ok(f"reward '{name}'")
        else:
            fail(f"reward '{name}'", f"not in registry {list(REWARD_REGISTRY)}")

    # ── 2. SFT datasets ────────────────────────────────────────
    if args.phase in ("sft", "both"):
        print("\n[SFT datasets]")
        for lvl in levels:
            d = Path(args.sft_data) / f"level_{lvl}"
            if not d.is_dir():
                fail(f"level_{lvl} dataset dir", f"{d}")
                continue
            # Arrow DatasetDict must have dataset_dict.json + train/ + validation/
            missing_parts = [
                p for p in ["dataset_dict.json", "train", "validation"]
                if not (d / p).exists()
            ]
            if missing_parts:
                fail(f"level_{lvl} dataset", f"missing {missing_parts} inside {d}")
            else:
                ok(f"level_{lvl}  ({d})")

    # ── 3. GRPO dataset ────────────────────────────────────────
    if args.phase in ("grpo", "both"):
        print("\n[GRPO dataset]")
        gp = Path(args.grpo_dataset)
        if not gp.is_file():
            fail(f"GRPO dataset", f"{gp}")
        else:
            try:
                with open(gp) as f:
                    rows = json.load(f)
                expected_cols = {"prompt", "ground_truth", "complexity", "level"}
                if rows and not expected_cols.issubset(rows[0].keys()):
                    fail("GRPO dataset columns", f"expected {expected_cols}, got {set(rows[0].keys())}")
                else:
                    counts = {}
                    for r in rows:
                        counts[r["level"]] = counts.get(r["level"], 0) + 1
                    summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                    ok(f"GRPO dataset  {len(rows)} rows  [{summary}]")
                    for lvl in levels:
                        lk = f"level {lvl}"
                        if counts.get(lk, 0) == 0:
                            fail(f"GRPO level {lvl}", f"0 rows found for '{lk}'")
            except json.JSONDecodeError as e:
                fail("GRPO dataset", f"invalid JSON — {e}")

    # ── 4. SFT adapters (grpo-only mode) ──────────────────────
    if args.phase == "grpo":
        print("\n[SFT adapters]")
        for lvl in levels:
            adapter = Path(args.sft_dir) / f"level_{lvl}" / "final_adapter"
            if adapter.is_dir() and (adapter / "adapter_config.json").exists():
                ok(f"level_{lvl} adapter  ({adapter})")
            else:
                fail(f"level_{lvl} adapter", f"{adapter} not found or incomplete")

    # ── 5. output dirs writable ────────────────────────────────
    print("\n[output dirs]")
    for label, path_str in [("SFT output", args.sft_dir), ("GRPO output", args.grpo_dir)]:
        p = Path(path_str)
        try:
            p.mkdir(parents=True, exist_ok=True)
            # probe write permission
            probe = p / ".preflight_probe"
            probe.touch()
            probe.unlink()
            ok(f"{label}  ({p})")
        except OSError as e:
            fail(f"{label} not writable", str(e))

    # ── 6. model reachable ─────────────────────────────────────
    print("\n[model]")
    try:
        AutoConfig.from_pretrained(args.model)
        ok(f"model config  ({args.model})")
    except Exception as e:
        fail(f"model '{args.model}'", str(e))

    # ── summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if errors:
        print(f"PREFLIGHT FAILED — {len(errors)} issue(s):")
        for e in errors:
            print(f"  • {e}")
        print("Fix the above before re-running.")
        print(f"{'='*60}\n")
        raise SystemExit(1)
    else:
        print("PREFLIGHT PASSED — all checks green, starting training.")
        print(f"{'='*60}\n")


def main():
    args   = parse_args()
    levels = [args.level] if args.level else sorted(LEVEL_LABELS.keys())

    print(f"\n{'='*60}")
    print(f"Gemma SFT + GRPO — model: {args.model}")
    print(f"Phase: {args.phase}   Levels: {levels}   Rewards: {args.rewards}")
    print(f"{'='*60}\n")

    preflight_check(args, levels)

    tokenizer = load_tokenizer(args.model)

    for level in levels:
        adapter_path = Path(args.sft_dir) / f"level_{level}" / "final_adapter"

        if args.phase in ("sft", "both"):
            adapter_path = train_sft_level(level, tokenizer, args)

        if args.phase in ("grpo", "both"):
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"SFT adapter not found: {adapter_path}\n"
                    f"Run with --phase sft first, or pass --sft-dir pointing to existing adapters."
                )
            train_grpo_level(level, tokenizer, adapter_path, args)

    print(f"\n[done] All levels complete.")
    print(f"  SFT adapters  → {args.sft_dir}/level_{{N}}/final_adapter")
    print(f"  GRPO adapters → {args.grpo_dir}/level_{{N}}/final_adapter")


if __name__ == "__main__":
    main()
