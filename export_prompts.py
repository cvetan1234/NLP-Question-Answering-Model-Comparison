import json
import argparse
from pathlib import Path

TEMPLATE = """You are an answer extractor.
Return ONLY the exact answer span from the context.
Do NOT add any extra words.

If the answer is not in the context, return exactly:
unknown

Context:
{context}

Question:
{question}

Answer:
"""

SEP = "\n" + ("=" * 80) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdb_json", required=True)
    ap.add_argument("--out_txt", default="results/tdb_chatgpt_prompts.txt")
    args = ap.parse_args()

    tdb = json.load(open(args.tdb_json, "r", encoding="utf-8"))
    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    for i, ex in enumerate(tdb, start=1):
        prompt = TEMPLATE.format(context=ex["context"], question=ex["question"])
        chunks.append(f"[{i}] id={ex['id']}\n{prompt}")

    out_path.write_text(SEP.join(chunks), encoding="utf-8")
    print(f"✅ Wrote: {out_path.resolve()} (N={len(tdb)})")

if __name__ == "__main__":
    main()
