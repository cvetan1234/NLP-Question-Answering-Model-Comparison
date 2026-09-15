#!/usr/bin/env python3
"""
eval_t5_checkpoints.py

Evaluate SQuAD-style EM/F1 for:
- each checkpoint-* under a T5 model directory
- and the final model directory itself

Works with your LOCAL flattened SQuAD JSON format:
[
  {
    "id": "...",
    "context": "...",
    "question": "...",
    "answers": {"text":[...], "answer_start":[...]}
  },
  ...
]

How it works (generative QA):
- Builds a prompt per example: "question: ...  context: ..."
- T5 generates an answer string
- Computes SQuAD EM/F1 vs gold answers (max over golds)

Run:
  nohup python eval_t5_checkpoints.py \
    --model_dir checkpoints/t5_squad_ft \
    --eval_json local_squad/td_a_validation.json \
    > eval_t5.log 2>&1 &
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm


# --------------------------
# SQuAD metrics (standard)
# --------------------------
def normalize_answer(s: str) -> str:
    """Lower, remove punctuation, articles and extra whitespace."""
    import string

    def lower(text): return text.lower()

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths) if ground_truths else metric_fn(prediction, "")


# --------------------------
# Data
# --------------------------
def load_flat_squad(path: str, max_examples: int = 0):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{p} does not contain a non-empty list.")
    if max_examples and max_examples > 0:
        data = data[:max_examples]
    for ex in data[:3]:
        assert "id" in ex and "context" in ex and "question" in ex and "answers" in ex
    return data


# --------------------------
# Prompt formatting (T5)
# --------------------------
def build_prompt(question: str, context: str) -> str:
    # Standard T5 QA prompt style used in many examples
    return f"question: {question}  context: {context}"


# --------------------------
# T5 generation
# --------------------------
@torch.inference_mode()
def predict_answers_t5(
    model,
    tokenizer,
    examples,
    device: str,
    input_max_length: int = 512,
    batch_size: int = 8,
    generation_max_new_tokens: int = 32,
    generation_max_length: int = 64,
    num_beams: int = 1,
):
    """
    Returns dict: id -> generated_answer_text

    Note:
    - Some older Transformers versions don't support generation_max_new_tokens in Trainer args.
      Here we call model.generate directly, so we can safely use max_new_tokens if available.
    - We include a fallback to max_length if max_new_tokens isn't supported.
    """
    ids = [ex["id"] for ex in examples]
    prompts = [build_prompt(ex["question"], ex["context"]) for ex in examples]

    model.to(device)
    model.eval()

    preds = {}

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating (T5)", leave=False):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_ids = ids[start:end]

        enc = tokenizer(
            batch_prompts,
            truncation=True,
            max_length=input_max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Prefer max_new_tokens; fallback to max_length for older generate() signatures.
        try:
            gen_ids = model.generate(
                **enc,
                max_new_tokens=generation_max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )
        except TypeError:
            gen_ids = model.generate(
                **enc,
                max_length=generation_max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for qid, text in zip(batch_ids, texts):
            preds[qid] = (text or "").strip()

    return preds


def eval_em_f1(preds_by_id, examples):
    total = len(examples)
    em_sum = 0.0
    f1_sum = 0.0

    for ex in examples:
        qid = ex["id"]
        pred = preds_by_id.get(qid, "")
        golds = ex.get("answers", {}).get("text", []) or [""]

        em_sum += metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1_sum += metric_max_over_ground_truths(f1_score, pred, golds)

    return {
        "n": total,
        "em": 100.0 * em_sum / total if total else 0.0,
        "f1": 100.0 * f1_sum / total if total else 0.0,
    }


# --------------------------
# Checkpoint discovery
# --------------------------
def list_checkpoints(model_dir: Path):
    ckpts = sorted(
        [p for p in model_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(re.findall(r"checkpoint-(\d+)", p.name)[0]) if re.findall(r"checkpoint-(\d+)", p.name) else 0
    )
    return ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="checkpoints/t5_squad_ft",
                    help="Directory that contains checkpoint-* folders and/or final model files")
    ap.add_argument("--eval_json", default="local_squad/td_a_validation.json")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = full; else first N examples (debug)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Input truncation for prompt
    ap.add_argument("--input_max_length", type=int, default=512)

    # Generation params
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gen_max_new_tokens", type=int, default=32)
    ap.add_argument("--gen_max_length", type=int, default=64)  # fallback if needed
    ap.add_argument("--num_beams", type=int, default=1)

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model_dir: {model_dir.resolve()}")

    examples = load_flat_squad(args.eval_json, max_examples=args.max_examples)

    candidates = list_checkpoints(model_dir)
    candidates.append(model_dir)  # final

    results = []
    for p in candidates:
        tag = p.name if p.name.startswith("checkpoint-") else "FINAL"
        print(f"\n=== Evaluating {tag} ===")

        tokenizer = AutoTokenizer.from_pretrained(p, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(p)

        preds = predict_answers_t5(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            device=args.device,
            input_max_length=args.input_max_length,
            batch_size=args.batch_size,
            generation_max_new_tokens=args.gen_max_new_tokens,
            generation_max_length=args.gen_max_length,
            num_beams=args.num_beams,
        )

        metrics = eval_em_f1(preds, examples)
        results.append((tag, metrics["em"], metrics["f1"], metrics["n"]))
        print(f"{tag}: EM={metrics['em']:.2f} | F1={metrics['f1']:.2f} | N={metrics['n']}")

    results_sorted = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)

    print("\n================= SUMMARY (ranked) =================")
    print(f"{'model':<16} {'EM':>8} {'F1':>8} {'N':>8}")
    for tag, em, f1, n in results_sorted:
        print(f"{tag:<16} {em:8.2f} {f1:8.2f} {n:8d}")

    best = results_sorted[0]
    print("\n✅ Best by (F1, EM):", best[0], f"(F1={best[2]:.2f}, EM={best[1]:.2f}, N={best[3]})")


if __name__ == "__main__":
    main()
