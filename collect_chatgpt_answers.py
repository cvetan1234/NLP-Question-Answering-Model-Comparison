#!/usr/bin/env python3
"""
collect_chatgpt_answers.py

Interactive script to collect ChatGPT answers for a SQuAD-style JSON dataset
and save them to:
  1) results/tdb_chatgpt_outputs.json  (raw answers, incrementally saved)
  2) results/tdb_chatgpt_clean.json    (clean format for evaluation)

Usage:
  python3 collect_chatgpt_answers.py --tdb_json local_squad/td_b_100.json
  python3 collect_chatgpt_answers.py --tdb_json local_squad/td_b_100.json --resume

Optional:
  --out_json results/tdb_chatgpt_outputs.json   (default)
"""

import json
import argparse
from pathlib import Path


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdb_json", required=True, help="Path to SQuAD-style JSON list with fields: id, question, context, ...")
    ap.add_argument("--out_json", default="results/tdb_chatgpt_outputs.json", help="Where to save raw answers JSON")
    ap.add_argument("--resume", action="store_true", help="Resume if out_json exists (skip already-answered ids)")
    args = ap.parse_args()

    # Load dataset
    tdb = json.load(open(args.tdb_json, "r", encoding="utf-8"))
    if not isinstance(tdb, list):
        raise ValueError("Expected --tdb_json to be a JSON LIST of examples.")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done = {}
    if args.resume and out_path.exists():
        prev = json.load(open(out_path, "r", encoding="utf-8"))
        if isinstance(prev, list):
            done = {x["id"]: x for x in prev if isinstance(x, dict) and "id" in x}
            print(f"Resuming: {len(done)} already answered.")
        else:
            print("Existing out_json is not a list. Ignoring resume file.")

    results = list(done.values())

    # Main loop
    for i, ex in enumerate(tdb, start=1):
        if not isinstance(ex, dict) or "id" not in ex or "question" not in ex:
            raise ValueError(f"Bad example at index {i-1}: expected dict with keys 'id' and 'question'.")

        qid = ex["id"]
        if qid in done:
            continue

        print("\n" + "=" * 80)
        print(f"[{i}/{len(tdb)}] id={qid}")
        print("QUESTION:", ex["question"])
        print("Paste the ChatGPT answer (single line). Then press Enter:")
        ans = input("> ").strip()

        item = {"id": qid, "answer_raw": ans}
        results.append(item)
        done[qid] = item

        # Incremental save after every answer
        write_json(out_path, results)
        print("saved.")

    print(f"\nDone. Saved raw: {out_path.resolve()} (N={len(results)})")

    # ===========================
    # Auto-create CLEAN JSON for evaluation
    # ===========================
    clean = [{"id": x["id"], "answer": x.get("answer_raw", "")} for x in results]
    clean_path = out_path.parent / "tdb_chatgpt_outputs.json"
    write_json(clean_path, clean)
    print(f"Saved clean: {clean_path.resolve()} (N={len(clean)})")


if __name__ == "__main__":
    main()
