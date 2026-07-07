"""
eval_gemma_vllm.py — fast GSM8K eval for Gemma using vLLM offline inference.

Workflow:
  1. Merge base + SFT adapter (+ GRPO adapter) into a full model and save it.
  2. Load the merged model with vLLM.
  3. Submit all 1319 GSM8K test prompts in one shot.

The merged model is cached at --merged_dir so the slow merge only runs once
per level. Delete that directory to force a re-merge.

Usage:
  python3 logic_inf/eval_gemma_vllm.py \
      --adapter checkpoints_gemma_sft/level_4/final_adapter \
      --level "level 4" --output eval_results_gemma_sft/l4.json

  python3 logic_inf/eval_gemma_vllm.py \
      --sft_adapter checkpoints_gemma_sft/level_4/final_adapter \
      --adapter     checkpoints_gemma_grpo/level_4/final_adapter \
      --level "level 4" --output eval_results_gemma_grpo/l4.json
"""

import argparse
import json
import os
import re
import shutil

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Gemma chat template — must match train_gemma_full.py
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


def _patch_peft_for_gemma4():
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


def _base_from_adapter_config(adapter_path: str) -> str:
    cfg = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg):
        return ""
    with open(cfg) as f:
        return json.load(f).get("base_model_name_or_path", "")


parser = argparse.ArgumentParser()
parser.add_argument("--model",       default="",  help="Base model (auto-detected from adapter_config if omitted)")
parser.add_argument("--sft_adapter", default="",  help="SFT adapter path (merged first for GRPO eval)")
parser.add_argument("--adapter",     default="",  help="Final adapter path (SFT-only or GRPO)")
parser.add_argument("--level",       required=True)
parser.add_argument("--output",      default="")
parser.add_argument("--merged_dir",  default="",  help="Where to cache the merged model. Default: <adapter_dir>/../merged")
parser.add_argument("--force_merge", action="store_true", help="Re-merge even if cached merged model exists")
parser.add_argument("--tensor_parallel", type=int, default=1, help="vLLM tensor parallel size (number of GPUs)")
args = parser.parse_args()

_detected = _base_from_adapter_config(args.sft_adapter or args.adapter)
if _detected and args.model and _detected != args.model:
    print(f"[WARN] --model {args.model} overridden by adapter base: {_detected}")
    args.model = _detected
elif _detected and not args.model:
    args.model = _detected
if not args.model:
    raise ValueError("Cannot determine base model: pass --model or a valid adapter path.")

_anchor = args.sft_adapter or args.adapter
if not args.merged_dir:
    args.merged_dir = os.path.join(os.path.dirname(_anchor), "merged_vllm")


def merge_model(model_name, sft_adapter, adapter, save_path):
    print(f"Merging: {model_name}")
    _patch_peft_for_gemma4()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    if sft_adapter:
        print(f"  + SFT adapter: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()
    if adapter:
        print(f"  + adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    print(f"Saving merged model → {save_path}")
    model.save_pretrained(save_path)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tok.chat_template:
        tok.chat_template = GEMMA_CHAT_TEMPLATE
    tok.save_pretrained(save_path)

    del model
    torch.cuda.empty_cache()


def build_prompt(question: str, level: str, tokenizer) -> str:
    label = GEMMA_LEVEL_LABELS.get(level, level)
    messages = [{"role": "user", "content": f"Solve this using {label}.\nProblem: {question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(response: str) -> str:
    m = re.search(r"###?#?\s*([\d\.,]+)", response)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"[\d\.,]+", response.replace(",", ""))
    return nums[-1].strip() if nums else ""


def main():
    level = args.level.lower().strip()
    style = LEVEL_STYLES.get(level, level)

    # ── 1. Merge (or reuse cached) ─────────────────────────────────────────────
    merged_dir = args.merged_dir
    if args.force_merge and os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)

    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir, exist_ok=True)
        try:
            merge_model(args.model, args.sft_adapter, args.adapter, merged_dir)
        except Exception:
            shutil.rmtree(merged_dir, ignore_errors=True)
            raise
    else:
        print(f"[SKIP merge] Using cached model at {merged_dir}")

    # ── 2. Load tokenizer ──────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, use_fast=True)
    if not tokenizer.chat_template:
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    # ── 3. Build all prompts ───────────────────────────────────────────────────
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    questions, gold_answers = [], []
    for row in dataset:
        m = re.search(r"####\s*([\d\.,]+)", row["answer"])
        if not m:
            continue
        questions.append(row["question"])
        gold_answers.append(m.group(1).strip().replace(",", ""))

    prompts = [build_prompt(q, level, tokenizer) for q in questions]
    print(f"Submitting {len(prompts)} prompts to vLLM ({level} / {style})...")

    # ── 4. vLLM inference ─────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=merged_dir,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["<end_of_turn>", "<eos>"],
    )
    outputs = llm.generate(prompts, sampling)

    # ── 5. Score ───────────────────────────────────────────────────────────────
    examples = []
    correct = total = 0
    total_think_tokens = total_think_chars = 0

    for i, out in enumerate(outputs):
        response = out.outputs[0].text
        predicted = extract_answer(response)
        is_correct = predicted == gold_answers[i]
        if is_correct:
            correct += 1
        total += 1

        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        think_content = think_match.group(1) if think_match else ""
        think_chars = len(think_content)
        think_tokens = len(tokenizer.encode(think_content, add_special_tokens=False)) if think_content else 0
        total_think_tokens += think_tokens
        total_think_chars += think_chars

        examples.append({
            "question":     questions[i],
            "gold":         gold_answers[i],
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
        "level":             level,
        "style":             style,
        "model":             args.model,
        "merged_dir":        merged_dir,
        "accuracy":          round(accuracy, 4),
        "correct":           correct,
        "total":             total,
        "mean_think_tokens": round(mean_think_tokens, 2),
        "mean_think_chars":  round(mean_think_chars,  2),
        "examples":          examples,
    }

    print(f"\nLevel:            {level} ({style})")
    print(f"Accuracy:         {accuracy:.4f} ({correct}/{total})")
    print(f"Mean Think Tokens:{mean_think_tokens:.2f}")
    print(f"Mean Think Chars: {mean_think_chars:.2f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
