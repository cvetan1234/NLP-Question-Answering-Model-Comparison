#!/usr/bin/env python3
"""
run_llama_answers.py

One-by-one LLaMA answer generator.
- Reads prompts in "Answer Extractor" format from tdb_chatgpt_prompts.txt
- Generates short exact answers
- Saves them in JSON after each prompt
- Test mode: first 2 prompts only
"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ===========================
# CONFIG
# ===========================
input_file = "prompts.txt"
output_file = "results/tdb_llama_outputs.json"
device = "cuda"                                  # GPU to use

Path("results").mkdir(parents=True, exist_ok=True)

# ===========================
# LOAD LLaMA-2-7B-Chat
# ===========================
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("✅ LLaMA-2-7B-Chat loaded successfully!")

# ===========================
# PARSE PROMPTS
# ===========================
prompts = []

with open(input_file, "r", encoding="utf-8") as f:
    current_id = None
    context_lines = []
    question_lines = []
    mode = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and "id=" in line:
            current_id = line.split("id=")[1].split("]")[0].strip()
        elif line.startswith("Context:"):
            mode = "context"
            context_lines.append(line[len("Context:"):].strip())
        elif line.startswith("Question:"):
            mode = "question"
            question_lines.append(line[len("Question:"):].strip())
        elif line.startswith("Answer:"):
            if current_id:
                context = " ".join(context_lines).strip()
                question = " ".join(question_lines).strip()
                prompts.append({
                    "id": current_id,
                    "context": context,
                    "question": question
                })
            current_id = None
            context_lines = []
            question_lines = []
            mode = None
        else:
            if mode == "context":
                context_lines.append(line)
            elif mode == "question":
                question_lines.append(line)

# ===========================
# GENERATE ANSWERS
# ===========================
results = []

for idx, ex in enumerate(prompts, 1):
    qid = ex["id"]

    prompt_text = f"""You are an answer extractor.
Return ONLY the exact answer span from the context.
Do NOT include any extra words, explanations, or repeated text.

Context:
{ex['context']}

Question:
{ex['question']}

Answer:
"""

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=0
    )

    # Decode output
    raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # === CLEAN OUTPUT ===
    # Take only text after last "Answer:" if present
    if "Answer:" in raw_answer:
        raw_answer = raw_answer.split("Answer:")[-1].strip()

    # Stop at first newline or unexpected phrases
    for sep in ["\n", "Expected output", "Actual output"]:
        if sep in raw_answer:
            raw_answer = raw_answer.split(sep)[0].strip()

    results.append({
        "id": qid,
        "answer": raw_answer
    })

    # Save after each prompt
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[{idx}/{len(prompts)}] Generated clean answer for id={qid}: {raw_answer}")

print(f"\n✅ Test run complete. Clean answers saved to {output_file}")
