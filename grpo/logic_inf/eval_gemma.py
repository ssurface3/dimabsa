"""
eval_gemma.py — GSM8K evaluator for Gemma models.

Differences from evaluate.py:
  - Patches Gemma chat template (tokenizer ships without one)
  - Prompt format matches train_gemma_full.py exactly
  - bfloat16 instead of float16
  - --sft_adapter for two-stage loading (base -> SFT merge -> GRPO LoRA)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Must match train_gemma_full.py exactly
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] | trim }}<end_of_turn>\n<start_of_turn>model\n"
    "{% if loop.last and add_generation_prompt %}<think>\n{% endif %}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] | trim }}<end_of_turn>\n"
    "{% endif %}"
    "{% endfor %}"
)

# Must match LEVEL_LABELS in train_gemma_full.py exactly
GEMMA_LEVEL_LABELS = {
    "level 1": "Level 1 (Verbose explanation)",
    "level 2": "Level 2 (Structured shorthand)",
    "level 3": "Level 3 (Variable chain)",
    "level 4": "Level 4 (Ultra-compact)",
    "level 5": "Level 5 (Pure expression)",
}

LEVEL_STYLES = {
    "level 1": "Verbose",
    "level 2": "Concise",
    "level 3": "Symbolic",
    "level 4": "Shorthand",
    "level 5": "Extreme",
}

import logging as _logging

def _patch_peft_for_gemma4():
    """
    Gemma 4 E4B's vision encoder wraps Linear in Gemma4ClippableLinear.
    PEFT tries to inject LoRA into every module matching target_modules —
    including vision encoder layers — and crashes on the custom type.
    Skip unsupported module types silently during injection.
    """
    try:
        from peft.tuners.lora.model import LoraModel
        _orig = LoraModel._create_and_replace
        def _patched(self, *args, **kwargs):
            try:
                _orig(self, *args, **kwargs)
            except ValueError as e:
                if "is not supported" in str(e):
                    return
                raise
        LoraModel._create_and_replace = _patched
    except Exception:
        pass

_patch_peft_for_gemma4()


def _load_adapter(model, adapter_path: str):
    """
    Load a PEFT adapter while suppressing the 'unexpected keys' warnings that
    come from vision_tower LoRA weights saved in the adapter but not injected
    (because we skip Gemma4ClippableLinear during injection).
    """
    peft_log = _logging.getLogger("peft")
    prev_level = peft_log.level
    peft_log.setLevel(_logging.ERROR)
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    finally:
        peft_log.setLevel(prev_level)
    return model


def _base_from_adapter_config(adapter_path: str) -> str:
    """Read base_model_name_or_path from the adapter's config so we always
    load the exact model architecture the adapter was trained on."""
    cfg = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg):
        return ""
    with open(cfg) as f:
        d = json.load(f)
    return d.get("base_model_name_or_path", "")


parser = argparse.ArgumentParser()
parser.add_argument("--model",       default="",     help="Base model name/path (auto-detected from adapter_config if omitted)")
parser.add_argument("--sft_adapter", default="",     help="SFT adapter path (merged before GRPO adapter)")
parser.add_argument("--adapter",     default="",     help="GRPO adapter path (or SFT adapter for sft-only eval)")
parser.add_argument("--level",       required=True,  help="e.g. 'level 4'")
parser.add_argument("--output",      default="",     help="Path to save JSON results")
parser.add_argument("--batch_size",  type=int, default=8)
args = parser.parse_args()

# Auto-detect base model from adapter config — avoids shape mismatch when the
# model ID drifts between training and eval.
_detected = _base_from_adapter_config(args.sft_adapter or args.adapter)
if _detected and args.model and _detected != args.model:
    print(f"[WARN] --model {args.model} differs from adapter's base ({_detected}).")
    print(f"[WARN] Using adapter's base model to avoid shape mismatch.")
    args.model = _detected
elif _detected and not args.model:
    args.model = _detected
if not args.model:
    raise ValueError("Cannot determine base model: pass --model or provide a valid adapter path.")

_NUM_RE = re.compile(r"[\d\.,]+")


def extract_answer(response: str) -> str:
    m = re.search(r"###?#?\s*([\d\.,]+)", response)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = _NUM_RE.findall(response.replace(",", ""))
    return nums[-1].strip() if nums else ""


def build_prompt(question: str, level: str) -> str:
    label = GEMMA_LEVEL_LABELS.get(level, level)
    return f"Solve this using {label}.\nProblem: {question}"


def main():
    level = args.level.lower().strip()
    style = LEVEL_STYLES.get(level, level)

    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = True   # needed for generation

    if args.sft_adapter:
        print(f"Merging SFT adapter: {args.sft_adapter}")
        model = _load_adapter(model, args.sft_adapter)
        model = model.merge_and_unload()

    if args.adapter:
        print(f"Merging adapter: {args.adapter}")
        model = _load_adapter(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    dataset = load_dataset("openai/gsm8k", "main")["test"]

    questions, gold_answers = [], []
    for row in dataset:
        m = re.search(r"####\s*([\d\.,]+)", row["answer"])
        if not m:
            continue
        questions.append(row["question"])
        gold_answers.append(m.group(1).strip().replace(",", ""))

    all_prompts = []
    for q in questions:
        prompt_content = build_prompt(q, level)
        all_prompts.append([{"role": "user", "content": prompt_content}])

    examples = []
    correct = 0
    total = 0
    total_think_tokens = 0
    total_think_chars = 0

    for batch_start in tqdm(
        range(0, len(all_prompts), args.batch_size),
        desc=f"{level} ({style})",
        total=(len(all_prompts) + args.batch_size - 1) // args.batch_size,
    ):
        batch_end = min(batch_start + args.batch_size, len(all_prompts))
        batch_messages  = all_prompts[batch_start:batch_end]
        batch_questions = questions[batch_start:batch_end]
        batch_golds     = gold_answers[batch_start:batch_end]

        batch_inputs = []
        for msgs in batch_messages:
            ids = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt"
            )
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            batch_inputs.append(ids.squeeze(0))

        padded = tokenizer.pad(
            {"input_ids": batch_inputs},
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **padded,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
            )

        input_len = padded["input_ids"].shape[1]
        for j in range(len(batch_questions)):
            generated_tokens = outputs[j][input_len:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

            predicted = extract_answer(response)
            is_correct = predicted == batch_golds[j]
            if is_correct:
                correct += 1
            total += 1

            # <think> is forced in the prompt; response starts from inside the block.
            # Handle both full <think>...</think> and completion-only </think>.
            m_full = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            m_tail = re.search(r"^(.*?)</think>", response, re.DOTALL)
            think_content = m_full.group(1) if m_full else (m_tail.group(1) if m_tail else "")
            think_tokens  = len(tokenizer.encode(think_content, add_special_tokens=False)) if think_content else 0
            think_chars   = len(think_content)
            total_think_tokens += think_tokens
            total_think_chars  += think_chars

            examples.append({
                "question":     batch_questions[j],
                "gold":         batch_golds[j],
                "predicted":    predicted,
                "correct":      is_correct,
                "think_tokens": think_tokens,
                "think_chars":  think_chars,
                "response":     response,
            })

    accuracy          = correct / total if total > 0 else 0.0
    mean_think_tokens = total_think_tokens / total if total > 0 else 0.0
    mean_think_chars  = total_think_chars  / total if total > 0 else 0.0

    results = {
        "level":            level,
        "style":            style,
        "model":            args.model,
        "sft_adapter":      args.sft_adapter,
        "adapter":          args.adapter,
        "accuracy":         round(accuracy, 4),
        "correct":          correct,
        "total":            total,
        "mean_think_tokens": round(mean_think_tokens, 2),
        "mean_think_chars":  round(mean_think_chars,  2),
        "examples":         examples,
    }

    print(f"\nLevel: {level} ({style})")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Mean Think Tokens: {mean_think_tokens:.2f}")
    print(f"Mean Think Chars:  {mean_think_chars:.2f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
