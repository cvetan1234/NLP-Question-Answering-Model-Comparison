# train_qa_local.py
# One script to fine-tune:
#   - DistilBERT (extractive QA)  -> AutoModelForQuestionAnswering
#   - T5         (generative QA)  -> AutoModelForSeq2SeqLM
#
# ✅ UPDATED (requested):
#    - Save checkpoint AFTER EACH EPOCH (not every N steps)
#    - Keep up to N epoch-checkpoints (default: keep all epochs)
#
# Run examples:
#   python train_qa_local.py --model distilbert --epochs 5 --out_dir checkpoints/distilbert_squad_ft
#   python train_qa_local.py --model t5 --epochs 5 --out_dir checkpoints/t5_squad_ft

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

DEFAULT_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "t5": "t5-small",
}


# ----------------------------
# Data loading
# ----------------------------
def load_local_json(path: str) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{p} does not contain a non-empty list.")
    return Dataset.from_list(data)


# ----------------------------
# DistilBERT (extractive) preprocessing
# ----------------------------
def prepare_features_extractive(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized["offset_mapping"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping", None)
    return tokenized


# ----------------------------
# T5 (generative) preprocessing
# ----------------------------
def prepare_features_t5(examples, tokenizer, max_input=512, max_target=32):
    inputs = []
    targets = []
    for q, c, ans in zip(examples["question"], examples["context"], examples["answers"]):
        inputs.append(f"question: {q}  context: {c}")
        targets.append(ans["text"][0] if ans and "text" in ans and len(ans["text"]) else "")

    model_inputs = tokenizer(
        inputs,
        max_length=max_input,
        truncation=True,
        padding="max_length",
    )

    # compatible with older transformers
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target,
            truncation=True,
            padding="max_length",
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ----------------------------
# Training: DistilBERT
# ----------------------------
def train_distilbert(
    base_model: str,
    train_ds: Dataset,
    eval_ds: Dataset,
    out_dir: str,
    epochs: float,
    lr: float,
    train_bs: int,
    eval_bs: int,
    max_length: int,
    doc_stride: int,
    eval_slice: int,
    keep_checkpoints: int | None,
):
    os.makedirs(out_dir, exist_ok=True)

    if eval_slice and eval_slice > 0:
        eval_ds = eval_ds.select(range(min(eval_slice, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(base_model)

    train_features = train_ds.map(
        lambda x: prepare_features_extractive(x, tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train (extractive)",
    )
    eval_features = eval_ds.map(
        lambda x: prepare_features_extractive(x, tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval (extractive)",
    )

    args = TrainingArguments(
        output_dir=out_dir,

        # ✅ save/eval each epoch
        evaluation_strategy="epoch",
        save_strategy="epoch",

        # ✅ keep last N checkpoints (or keep all if None)
        save_total_limit=keep_checkpoints,

        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_features,
        eval_dataset=eval_features,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(out_dir)          # final (last) model
    tokenizer.save_pretrained(out_dir)

    print(f"\n✅ Saved fine-tuned DistilBERT to: {out_dir}")


# ----------------------------
# Training: T5
# ----------------------------
def train_t5(
    base_model: str,
    train_ds: Dataset,
    eval_ds: Dataset,
    out_dir: str,
    epochs: float,
    lr: float,
    train_bs: int,
    eval_bs: int,
    max_input: int,
    max_target: int,
    eval_slice: int,
    keep_checkpoints: int | None,
):
    os.makedirs(out_dir, exist_ok=True)

    if eval_slice and eval_slice > 0:
        eval_ds = eval_ds.select(range(min(eval_slice, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    train_features = train_ds.map(
        lambda x: prepare_features_t5(x, tokenizer, max_input=max_input, max_target=max_target),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train (t5)",
    )
    eval_features = eval_ds.map(
        lambda x: prepare_features_t5(x, tokenizer, max_input=max_input, max_target=max_target),
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval (t5)",
    )

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,

        # ✅ save/eval each epoch
        evaluation_strategy="epoch",
        save_strategy="epoch",

        # ✅ keep last N checkpoints (or keep all if None)
        save_total_limit=keep_checkpoints,

        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_features,
        eval_dataset=eval_features,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(out_dir)          # final (last) model
    tokenizer.save_pretrained(out_dir)

    print(f"\n✅ Saved fine-tuned T5 to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["distilbert", "t5"], required=True)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--train_json", default="local_squad/train.json")
    ap.add_argument("--eval_json", default="local_squad/td_a_validation.json")
    ap.add_argument("--out_dir", default=None)

    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)

    ap.add_argument("--train_bs", type=int, default=4)
    ap.add_argument("--eval_bs", type=int, default=4)

    ap.add_argument("--eval_slice", type=int, default=2000)

    # ✅ new: checkpoint retention
    ap.add_argument(
        "--keep_checkpoints",
        type=int,
        default=0,
        help="How many epoch checkpoints to keep. 0 = keep all (recommended for your project).",
    )

    # DistilBERT-specific
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)

    # T5-specific
    ap.add_argument("--max_input", type=int, default=512)
    ap.add_argument("--max_target", type=int, default=32)

    args = ap.parse_args()

    train_ds = load_local_json(args.train_json)
    eval_ds = load_local_json(args.eval_json)

    base_model = args.base_model or DEFAULT_MODELS[args.model]
    out_dir = args.out_dir or f"checkpoints/{args.model}_squad_ft"

    # HF expects None to mean "no limit"
    keep = None if args.keep_checkpoints == 0 else args.keep_checkpoints

    if args.model == "distilbert":
        train_distilbert(
            base_model=base_model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            train_bs=args.train_bs,
            eval_bs=args.eval_bs,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
            eval_slice=args.eval_slice,
            keep_checkpoints=keep,
        )
    else:
        train_t5(
            base_model=base_model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            train_bs=args.train_bs,
            eval_bs=args.eval_bs,
            max_input=args.max_input,
            max_target=args.max_target,
            eval_slice=args.eval_slice,
            keep_checkpoints=keep,
        )

    print(f"Train file: {Path(args.train_json).resolve()}")
    print(f"Eval  file: {Path(args.eval_json).resolve()} (slice={args.eval_slice if args.eval_slice else 'FULL'})")
    print(f"Out   dir : {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()
