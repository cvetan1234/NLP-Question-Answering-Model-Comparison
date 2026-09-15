#!/usr/bin/env python3
"""
compare_4_models_on_tdb.py

Run:
  python compare_4_models_on_tdb.py

Hard-coded defaults below:
  - TD-B gold JSON
  - DistilBERT fine-tuned checkpoint folder
  - T5 fine-tuned checkpoint folder
  - ChatGPT predictions JSON
  - LLaMA predictions JSON

It will compute SQuAD EM/F1 for all four and save:
  results/tdb_compare_4models.json
"""

import re
import json
from pathlib import Path
from collections import Counter

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

# ============================================================
# ✅ HARD-CODED PATHS (YOUR REAL ONES)
# ============================================================
TDB_JSON        = Path(r"local_squad/td_b_100.json")

DISTILBERT_PATH = Path(r"C:\Users\cveta\Desktop\Project NLP 2026\FINAL 2\best_checkpoint_distillbert")
T5_PATH         = Path(r"C:\Users\cveta\Desktop\Project NLP 2026\FINAL 2\best_checkpoint_t5")

CHATGPT_RAW     = Path(r"results/tdb_chatgpt_outputs.json")
LLAMA_RAW       = Path(r"results/tdb_llama_outputs.json")

OUT_DIR         = Path("results")
OUT_JSON        = OUT_DIR / "tdb_compare_4models.json"

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# DistilBERT decoding params
DB_MAX_LENGTH         = 384
DB_DOC_STRIDE         = 128
DB_MAX_ANSWER_LENGTH  = 30
DB_BATCH_SIZE         = 16
DB_TOP_K              = 20

# T5 decoding params
T5_INPUT_MAX_LENGTH   = 512
T5_BATCH_SIZE         = 8
T5_GEN_MAX_NEW_TOKENS = 32
T5_GEN_MAX_LENGTH     = 64
T5_NUM_BEAMS          = 1


# ----------------------------
# SQuAD normalization + metrics
# ----------------------------
def normalize_answer(s: str) -> str:
    import string
    def lower(text): return text.lower()
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))

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
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, gt) for gt in ground_truths)

def eval_em_f1(preds_by_id, examples):
    total = len(examples)
    em_sum = 0.0
    f1_sum = 0.0
    missing = 0

    for ex in examples:
        qid = ex["id"]
        pred = preds_by_id.get(qid, "")
        if qid not in preds_by_id:
            missing += 1
        golds = ex.get("answers", {}).get("text", []) or [""]

        em_sum += metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1_sum += metric_max_over_ground_truths(f1_score, pred, golds)

    return {
        "n": total,
        "missing": missing,
        "em": 100.0 * em_sum / total if total else 0.0,
        "f1": 100.0 * f1_sum / total if total else 0.0,
    }


# ----------------------------
# IO helpers
# ----------------------------
def load_list_json(path: Path):
    p = path.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{p} must be a non-empty list JSON.")
    return data

def load_flat_squad(path: Path):
    data = load_list_json(path)
    for ex in data[:3]:
        if not all(k in ex for k in ["id", "context", "question", "answers"]):
            raise ValueError(f"{path} does not look like flat SQuAD JSON.")
    return data

def build_pred_map(pred_rows):
    """
    Accepts either:
      - {"id":..., "answer_raw":...}
      - {"id":..., "answer":...}
    Prefers "answer" if present.
    """
    preds = {}
    skipped = 0
    for r in pred_rows:
        if not isinstance(r, dict) or "id" not in r:
            skipped += 1
            continue
        if "answer" in r and r["answer"] is not None:
            ans = str(r["answer"])
        else:
            ans = str(r.get("answer_raw", "") or "")
        preds[str(r["id"])] = ans
    return preds, skipped


# ----------------------------
# DistilBERT decoding (extractive QA)
# ----------------------------
@torch.inference_mode()
def predict_answers_distilbert(
    model,
    tokenizer,
    examples,
    device: str,
    max_length: int,
    doc_stride: int,
    max_answer_length: int,
    batch_size: int,
    top_k: int,
):
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
    )
    overflow_to_sample = enc["overflow_to_sample_mapping"]
    offset_mapping = enc["offset_mapping"]

    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

    model.to(device).eval()
    best = {qid: {"score": -1e18, "text": ""} for qid in ids}

    nfeat = input_ids.size(0)
    for start in tqdm(range(0, nfeat, batch_size), desc="DistilBERT QA", leave=False):
        end = min(start + batch_size, nfeat)
        batch = {
            "input_ids": input_ids[start:end].to(device),
            "attention_mask": attention_mask[start:end].to(device),
        }
        out = model(**batch)
        s_logits = out.start_logits.detach().cpu()
        e_logits = out.end_logits.detach().cpu()

        for i in range(end - start):
            feature_idx = start + i
            sample_idx = int(overflow_to_sample[feature_idx])
            qid = ids[sample_idx]

            offsets = offset_mapping[feature_idx]
            seq_ids = enc.sequence_ids(feature_idx)
            context_token_indices = [j for j, s in enumerate(seq_ids) if s == 1]
            if not context_token_indices:
                continue

            s_log = s_logits[i].tolist()
            e_log = e_logits[i].tolist()

            start_c = sorted(range(len(s_log)), key=lambda j: s_log[j], reverse=True)[:top_k]
            end_c = sorted(range(len(e_log)), key=lambda j: e_log[j], reverse=True)[:top_k]

            for s_idx in start_c:
                for e_idx in end_c:
                    if s_idx not in context_token_indices or e_idx not in context_token_indices:
                        continue
                    if e_idx < s_idx:
                        continue
                    if (e_idx - s_idx + 1) > max_answer_length:
                        continue

                    s_off = offsets[s_idx]
                    e_off = offsets[e_idx]
                    if s_off is None or e_off is None:
                        continue

                    char_s = s_off[0]
                    char_e = e_off[1]
                    if char_s is None or char_e is None or char_e <= char_s:
                        continue

                    text = contexts[sample_idx][char_s:char_e]
                    score = s_log[s_idx] + e_log[e_idx]
                    if score > best[qid]["score"]:
                        best[qid] = {"score": score, "text": text}

    return {qid: best[qid]["text"] for qid in best}


# ----------------------------
# T5 decoding (generative QA)
# ----------------------------
def build_t5_prompt(q: str, c: str) -> str:
    return f"question: {q}  context: {c}"

@torch.inference_mode()
def predict_answers_t5(
    model,
    tokenizer,
    examples,
    device: str,
    input_max_length: int,
    batch_size: int,
    gen_max_new_tokens: int,
    gen_max_length: int,
    num_beams: int,
):
    ids = [ex["id"] for ex in examples]
    prompts = [build_t5_prompt(ex["question"], ex["context"]) for ex in examples]

    model.to(device).eval()
    preds = {}

    for start in tqdm(range(0, len(prompts), batch_size), desc="T5 generate", leave=False):
        end = min(start + batch_size, len(prompts))
        enc = tokenizer(
            prompts[start:end],
            truncation=True,
            max_length=input_max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        early_stopping = True if num_beams > 1 else False

        # compatibility with older Transformers
        try:
            gen_ids = model.generate(
                **enc,
                max_new_tokens=gen_max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping,
            )
        except TypeError:
            gen_ids = model.generate(
                **enc,
                max_length=gen_max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
            )

        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for qid, txt in zip(ids[start:end], texts):
            preds[qid] = (txt or "").strip()

    return preds


# ----------------------------
# Pretty printing
# ----------------------------
def print_table(rows):
    headers = ["Model", "EM", "F1", "Missing", "Notes"]

    w_model = max(len(headers[0]), max(len(r["model"]) for r in rows))
    w_em = max(len(headers[1]), 6)
    w_f1 = max(len(headers[2]), 6)
    w_miss = max(len(headers[3]), 7)
    w_notes = max(len(headers[4]), max(len(r.get("notes", "")) for r in rows))

    def line():
        print(f"+-{'-'*w_model}-+-{'-'*w_em}-+-{'-'*w_f1}-+-{'-'*w_miss}-+-{'-'*w_notes}-+")

    line()
    print(f"| {headers[0]:<{w_model}} | {headers[1]:>{w_em}} | {headers[2]:>{w_f1}} | {headers[3]:>{w_miss}} | {headers[4]:<{w_notes}} |")
    line()
    for r in rows:
        print(f"| {r['model']:<{w_model}} | {r['em']:>{w_em}.2f} | {r['f1']:>{w_f1}.2f} | {r['missing']:>{w_miss}d} | {r.get('notes',''):<{w_notes}} |")
    line()


def main():
    # ---- sanity checks ----
    for p in [TDB_JSON, DISTILBERT_PATH, T5_PATH, CHATGPT_RAW, LLAMA_RAW]:
        if not p.exists():
            raise FileNotFoundError(f"❌ Missing path: {p.resolve()}")

    print(f"Gold TD-B: {TDB_JSON.resolve()}")
    print(f"Device:   {DEVICE}")

    examples = load_flat_squad(TDB_JSON)
    print(f"N examples: {len(examples)}")

    # 1) ChatGPT file
    chat_rows = load_list_json(CHATGPT_RAW)
    chat_map, chat_skipped = build_pred_map(chat_rows)
    chat_metrics = eval_em_f1(chat_map, examples)

    # 2) LLaMA file
    llama_rows = load_list_json(LLAMA_RAW)
    llama_map, llama_skipped = build_pred_map(llama_rows)
    llama_metrics = eval_em_f1(llama_map, examples)

    # 3) DistilBERT inference
    print("\nLoading DistilBERT...")
    db_tok = AutoTokenizer.from_pretrained(DISTILBERT_PATH, use_fast=True)
    db_model = AutoModelForQuestionAnswering.from_pretrained(DISTILBERT_PATH)

    db_preds = predict_answers_distilbert(
        db_model, db_tok, examples, DEVICE,
        max_length=DB_MAX_LENGTH,
        doc_stride=DB_DOC_STRIDE,
        max_answer_length=DB_MAX_ANSWER_LENGTH,
        batch_size=DB_BATCH_SIZE,
        top_k=DB_TOP_K,
    )
    db_metrics = eval_em_f1(db_preds, examples)

    # 4) T5 inference
    print("\nLoading T5...")
    t5_tok = AutoTokenizer.from_pretrained(T5_PATH, use_fast=True)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_PATH)

    t5_preds = predict_answers_t5(
        t5_model, t5_tok, examples, DEVICE,
        input_max_length=T5_INPUT_MAX_LENGTH,
        batch_size=T5_BATCH_SIZE,
        gen_max_new_tokens=T5_GEN_MAX_NEW_TOKENS,
        gen_max_length=T5_GEN_MAX_LENGTH,
        num_beams=T5_NUM_BEAMS,
    )
    t5_metrics = eval_em_f1(t5_preds, examples)

    # Summary table
    print("\n==================== RESULTS (TD-B) ====================")
    rows = [
        {
            "model": "ChatGPT (file)",
            "em": chat_metrics["em"],
            "f1": chat_metrics["f1"],
            "missing": chat_metrics["missing"],
            "notes": (f"skipped_rows={chat_skipped}" if chat_skipped else ""),
        },
        {
            "model": "LLaMA (file)",
            "em": llama_metrics["em"],
            "f1": llama_metrics["f1"],
            "missing": llama_metrics["missing"],
            "notes": (f"skipped_rows={llama_skipped}" if llama_skipped else ""),
        },
        {
            "model": "DistilBERT (ft)",
            "em": db_metrics["em"],
            "f1": db_metrics["f1"],
            "missing": db_metrics["missing"],
            "notes": f"bs={DB_BATCH_SIZE}",
        },
        {
            "model": "T5 (ft)",
            "em": t5_metrics["em"],
            "f1": t5_metrics["f1"],
            "missing": t5_metrics["missing"],
            "notes": f"beams={T5_NUM_BEAMS}",
        },
    ]
    print_table(rows)

    # Deltas vs ChatGPT
    def delta(m):
        return {"d_em": m["em"] - chat_metrics["em"], "d_f1": m["f1"] - chat_metrics["f1"]}

    deltas = {
        "llama_minus_chatgpt": delta(llama_metrics),
        "distilbert_minus_chatgpt": delta(db_metrics),
        "t5_minus_chatgpt": delta(t5_metrics),
    }

    print("\nDeltas vs ChatGPT (model - ChatGPT):")
    print(f"  LLaMA:      ΔEM={deltas['llama_minus_chatgpt']['d_em']:+.2f} | ΔF1={deltas['llama_minus_chatgpt']['d_f1']:+.2f}")
    print(f"  DistilBERT: ΔEM={deltas['distilbert_minus_chatgpt']['d_em']:+.2f} | ΔF1={deltas['distilbert_minus_chatgpt']['d_f1']:+.2f}")
    print(f"  T5:         ΔEM={deltas['t5_minus_chatgpt']['d_em']:+.2f} | ΔF1={deltas['t5_minus_chatgpt']['d_f1']:+.2f}")

    # Save everything
    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "tdb_json": str(TDB_JSON.resolve()),
        "device": DEVICE,
        "inputs": {
            "distilbert_path": str(DISTILBERT_PATH.resolve()),
            "t5_path": str(T5_PATH.resolve()),
            "chatgpt_raw": str(CHATGPT_RAW.resolve()),
            "llama_raw": str(LLAMA_RAW.resolve()),
        },
        "metrics": {
            "chatgpt": chat_metrics,
            "llama": llama_metrics,
            "distilbert": db_metrics,
            "t5": t5_metrics,
        },
        "deltas_vs_chatgpt": deltas,
        "preds": {
            "chatgpt": chat_map,
            "llama": llama_map,
            "distilbert": db_preds,
            "t5": t5_preds,
        },
        "skipped_rows": {
            "chatgpt": chat_skipped,
            "llama": llama_skipped,
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved full comparison (metrics + predictions) to: {OUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
