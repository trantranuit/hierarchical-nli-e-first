"""Pull the real ViNLI dataset (CSV mirror at trantranuit/ViHLM_NLI_Project on GitHub,
data/vinli/{train,dev,test}.csv — public, no auth/gating) and convert it into
data/raw/{train,dev,test}.jsonl matching this repo's schema:
    {"id": str, "premise": str, "hypothesis": str, "label": "E"|"C"|"N"}

Source CSV schema: sentence1,sentence2,label (label is an int: 0=entailment,
1=neutral, 2=contradiction — per src/dataset.py's default label_map in that repo).
This script only renames/remaps fields, it does not touch the actual pairs.
Overwrites data/raw/{train,dev,test}.jsonl (same target files as prepare_vianli.py —
run only one of the two prepare scripts depending on which dataset you want to train on).

Usage:
    python3 scripts/prepare_vinli.py --config configs/config_vinli.yaml
"""
from __future__ import annotations
import argparse, json, pathlib
import yaml
import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/trantranuit/ViHLM_NLI_Project/main/data/vinli"
SPLIT_FILES = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
LABEL_MAP = {0: "E", 1: "N", 2: "C"}  # entailment=0, neutral=1, contradiction=2


def convert_split(split: str, out_path: pathlib.Path):
    url = f"{RAW_BASE}/{SPLIT_FILES[split]}"
    df = pd.read_csv(url)
    rows = []
    for idx, r in df.iterrows():
        label = LABEL_MAP.get(int(r["label"]))
        if label is None:
            print(f"[prepare_vinli] skip unknown label {r['label']!r} (row {idx})")
            continue
        rows.append({"id": f"vinli-{split}-{idx:06d}", "premise": str(r["sentence1"]),
                     "hypothesis": str(r["sentence2"]), "label": label})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = {}
    for r in rows:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    print(f"[prepare_vinli] {split}: {len(rows)} pairs -> {out_path} | label dist={dist}")
    return len(rows)


def main(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg = cfg["data"]
    raw_dir = pathlib.Path(data_cfg["raw_dir"])
    total = 0
    for split, key in [("train", "train_file"), ("dev", "dev_file"), ("test", "test_file")]:
        out_path = pathlib.Path(data_cfg[key])
        total += convert_split(split, out_path)
    print(f"[prepare_vinli] done — {total} pairs total -> {raw_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_vinli.yaml")
    args = ap.parse_args()
    main(args)
