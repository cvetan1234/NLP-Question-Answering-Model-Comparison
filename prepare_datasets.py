# build_local_squad_no_hf_with_starts.py
# Self-sufficient SQuAD v1.1 downloader + local TD-A / TD-B builder
# Includes answers.text + answers.answer_start (needed for DistilBERT extractive QA)
# NO HuggingFace "datasets" required.

import json
import random
from pathlib import Path
from urllib.request import urlopen, Request

OUT_DIR = Path("local_squad")
TD_B_SIZE = 100
SEED = 42

# Official SQuAD v1.1 release files (JSON)
SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQUAD_DEV_URL   = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def download_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def flatten_squad_keep_offsets(squad_json: dict) -> list:
    """
    Convert nested SQuAD structure into flat examples compatible with
    HuggingFace SQuAD schema:
      {
        "id": str,
        "context": str,
        "question": str,
        "answers": {
            "text": [str, ...],
            "answer_start": [int, ...]
        }
      }
    """
    out = []
    for article in squad_json.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qid = qa.get("id", "")
                question = qa.get("question", "")

                texts = []
                starts = []
                for a in qa.get("answers", []):
                    t = a.get("text", "")
                    s = a.get("answer_start", None)
                    if t != "" and isinstance(s, int):
                        texts.append(t)
                        starts.append(s)

                # SQuAD v1.1 should always have answers, but keep safe defaults
                if not texts:
                    texts = [""]
                    starts = [0]

                out.append({
                    "id": qid,
                    "context": context,
                    "question": question,
                    "answers": {
                        "text": texts,
                        "answer_start": starts
                    }
                })
    return out


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Downloading SQuAD v1.1 (official JSON) ...")
    train_raw = download_json(SQUAD_TRAIN_URL)
    dev_raw   = download_json(SQUAD_DEV_URL)

    print("Flattening dataset (keeping answer_start offsets) ...")
    train = flatten_squad_keep_offsets(train_raw)
    td_a  = flatten_squad_keep_offsets(dev_raw)  # TD-A = full dev set

    print(f"Train samples: {len(train)}")
    print(f"TD-A samples  : {len(td_a)}")

    random.seed(SEED)
    td_b = random.sample(td_a, TD_B_SIZE)

    save_json(OUT_DIR / "train.json", train)
    save_json(OUT_DIR / "td_a_validation.json", td_a)
    save_json(OUT_DIR / "td_b_100.json", td_b)

    # Optional: keep raw originals too
    raw_dir = OUT_DIR / "raw"
    save_json(raw_dir / "train-v1.1.json", train_raw)
    save_json(raw_dir / "dev-v1.1.json", dev_raw)

    print("\nSaved local dataset:")
    print(f" - {OUT_DIR / 'train.json'}")
    print(f" - {OUT_DIR / 'td_a_validation.json'}")
    print(f" - {OUT_DIR / 'td_b_100.json'}")
    print("\n✅ Now your DistilBERT training script can use answers['text'] and answers['answer_start'].")


if __name__ == "__main__":
    main()
