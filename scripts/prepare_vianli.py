"""Pull the real ViANLI dataset (uitnlp/ViANLI on HF Hub) and convert it into
data/raw/{train,dev,test}.jsonl matching this repo's schema:
    {"id": str, "premise": str, "hypothesis": str, "label": "E"|"C"|"N"}

ViANLI ships as {"uid","premise","hypothesis","label"} with label in
{"entailment","contradiction","neutral"} — this script only renames/remaps fields,
it does not touch the actual pairs. Overwrites the synthetic placeholder files that
ship in this repo for local --mock testing.

Usage:
    python3 scripts/prepare_vianli.py --config configs/config.yaml
"""
from __future__ import annotations
import argparse, json, pathlib
import yaml
from huggingface_hub import hf_hub_download

HF_REPO = "uitnlp/ViANLI"
SPLIT_FILES = {"train": "vianli_train.jsonl", "dev": "vianli_dev.jsonl", "test": "vianli_test.jsonl"}
LABEL_MAP = {"entailment": "E", "contradiction": "C", "neutral": "N"}


def convert_split(split: str, out_path: pathlib.Path):
    src_path = hf_hub_download(repo_id=HF_REPO, filename=SPLIT_FILES[split], repo_type="dataset")
    rows = []
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            label = LABEL_MAP.get(r["label"])
            if label is None:
                print(f"[prepare_vianli] skip unknown label {r['label']!r} (uid={r.get('uid')})")
                continue
            rows.append({"id": r["uid"], "premise": r["premise"], "hypothesis": r["hypothesis"], "label": label})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = {}
    for r in rows:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    print(f"[prepare_vianli] {split}: {len(rows)} pairs -> {out_path} | label dist={dist}")
    return len(rows)


def main(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg = cfg["data"]
    raw_dir = pathlib.Path(data_cfg["raw_dir"])
    total = 0
    for split, key in [("train", "train_file"), ("dev", "dev_file"), ("test", "test_file")]:
        out_path = pathlib.Path(data_cfg[key])
        total += convert_split(split, out_path)
    print(f"[prepare_vianli] done — {total} pairs total from {HF_REPO} -> {raw_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args)
