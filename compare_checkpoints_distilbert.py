#!/usr/bin/env python3
"""
eval_distilbert_checkpoints.py

Evaluate SQuAD-style EM/F1 for:
- each checkpoint-* under a model directory
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
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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
    # sanity
    for ex in data[:3]:
        assert "id" in ex and "context" in ex and "question" in ex and "answers" in ex
    return data


# --------------------------
# QA decoding (windowed)
# --------------------------
@torch.inference_mode()
def predict_answers(
    model,
    tokenizer,
    examples,
    device: str,
    max_length: int = 384,
    doc_stride: int = 128,
    max_answer_length: int = 30,
    batch_size: int = 16,
):
    """
    Returns dict: id -> predicted_answer_text

    Implements standard HF QA decoding:
    - tokenize with overflow + offsets
    - run model on features
    - pick best span per example across all windows

    IMPORTANT:
    - sequence_ids() is a method on the BatchEncoding returned by tokenizer (enc),
      not a method of the tokenizer itself.

    IMPORTANT FIX (your F1 regression):
    - span char range must use:
        char_s = offsets[s_idx][0]
        char_e = offsets[e_idx][1]
      NOT offsets[s_idx] for both.
    """
    questions = [ex["question"] for ex in examples]
    contexts = [ex["context"] for ex in examples]
    ids = [ex["id"] for ex in examples]

    enc = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=None,
    )

    overflow_to_sample = enc["overflow_to_sample_mapping"]
    offset_mapping = enc["offset_mapping"]

    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

    model.to(device)
    model.eval()

    best = {qid: {"score": -1e18, "text": ""} for qid in ids}

    n = input_ids.size(0)
    for start in tqdm(range(0, n, batch_size), desc="Running QA", leave=False):
        end = min(start + batch_size, n)
        batch = {
            "input_ids": input_ids[start:end].to(device),
            "attention_mask": attention_mask[start:end].to(device),
        }
        outputs = model(**batch)
        start_logits = outputs.start_logits.detach().cpu()  # (b, L)
        end_logits = outputs.end_logits.detach().cpu()      # (b, L)

        for i in range(end - start):
            feature_idx = start + i
            sample_idx = int(overflow_to_sample[feature_idx])
            qid = ids[sample_idx]

            offsets = offset_mapping[feature_idx]
            seq_ids = enc.sequence_ids(feature_idx)

            # context tokens are those with sequence_id == 1
            context_token_indices = [j for j, s in enumerate(seq_ids) if s == 1]
            if not context_token_indices:
                continue

            s_log = start_logits[i].tolist()
            e_log = end_logits[i].tolist()

            top_k = 20
            start_candidates = sorted(range(len(s_log)), key=lambda j: s_log[j], reverse=True)[:top_k]
            end_candidates = sorted(range(len(e_log)), key=lambda j: e_log[j], reverse=True)[:top_k]

            for s_idx in start_candidates:
                for e_idx in end_candidates:
                    if s_idx not in context_token_indices or e_idx not in context_token_indices:
                        continue
                    if e_idx < s_idx:
                        continue
                    if (e_idx - s_idx + 1) > max_answer_length:
                        continue

                    # offsets are (char_start, char_end) in the *context string*
                    # Some tokens can have (0,0) or None-like values depending on tokenizer;
                    # we skip invalid ones.
                    s_off = offsets[s_idx]
                    e_off = offsets[e_idx]
                    if s_off is None or e_off is None:
                        continue

                    char_s = s_off[0]
                    char_e = e_off[1]

                    if char_s is None or char_e is None:
                        continue
                    if char_e <= char_s:
                        continue

                    text = contexts[sample_idx][char_s:char_e]
                    score = s_log[s_idx] + e_log[e_idx]

                    if score > best[qid]["score"]:
                        best[qid] = {"score": score, "text": text}

    return {qid: best[qid]["text"] for qid in best}


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
    ap.add_argument("--model_dir", default="checkpoints/distilbert_squad_ft",
                    help="Directory that contains checkpoint-* folders and/or final model files")
    ap.add_argument("--eval_json", default="local_squad/td_a_validation.json")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = full TD-A; else first N examples (debug)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--max_answer_length", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
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
        model = AutoModelForQuestionAnswering.from_pretrained(p)

        preds = predict_answers(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            device=args.device,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
            max_answer_length=args.max_answer_length,
            batch_size=args.batch_size,
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
