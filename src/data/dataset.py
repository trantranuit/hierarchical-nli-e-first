"""Dataset + synthetic generator — Hierarchical NLI E-first."""
from __future__ import annotations
import json, random, pathlib
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    Dataset = object  # fallback for mock/offline

LABELS_FLAT = ["E", "C", "N"]
LABEL2ID_FLAT = {l:i for i,l in enumerate(LABELS_FLAT)}
ID2LABEL_FLAT = {i:l for l,i in LABEL2ID_FLAT.items()}

# Hierarchical mapping — 2 head_design hỗ trợ song song (Run A = e_first, Run B = n_first)
# Gold -> (coarse_label, fine_label or None)
HEAD_DESIGNS = {
    "e_first": {
        "gold_to_coarse": {"E": 0, "C": 1, "N": 1},        # E=0, Non-E=1
        "gold_to_fine":   {"E": None, "C": 0, "N": 1},     # C=0, N=1, E=ignore (-100)
        "coarse_labels": ["E", "Non-E"],
        "fine_labels":   ["C", "N"],
    },
    "n_first": {
        "gold_to_coarse": {"N": 0, "E": 1, "C": 1},        # N=0, Non-N=1
        "gold_to_fine":   {"N": None, "E": 0, "C": 1},     # E=0, C=1, N=ignore (-100)
        "coarse_labels": ["N", "Non-N"],
        "fine_labels":   ["E", "C"],
    },
}
# giữ tên cũ trỏ về e_first mặc định — không có nơi nào khác trong repo import các biến này
GOLD_TO_COARSE = HEAD_DESIGNS["e_first"]["gold_to_coarse"]
GOLD_TO_FINE   = HEAD_DESIGNS["e_first"]["gold_to_fine"]
COARSE_LABELS  = HEAD_DESIGNS["e_first"]["coarse_labels"]
FINE_LABELS    = HEAD_DESIGNS["e_first"]["fine_labels"]

VI_TEMPLATES = {
    "E": [
        ("Người đàn ông đang chạy.", "Một người đang chạy."),
        ("Trời đang mưa rất to.", "Thời tiết có mưa."),
        ("Cô ấy mua một chiếc váy đỏ.", "Cô ấy mua váy."),
        ("Hà Nội là thủ đô của Việt Nam.", "Hà Nội thuộc Việt Nam."),
        ("Mèo đang nằm trên ghế sofa.", "Có một con mèo trên ghế."),
    ],
    "C": [
        ("Trời đang mưa.", "Trời đang nắng."),
        ("Anh ấy còn sống.", "Anh ấy đã chết."),
        ("Cửa đang mở.", "Cửa đang đóng."),
        ("Cô ấy thích ăn chay.", "Cô ấy thích ăn thịt mỗi ngày."),
        ("Đội A thắng trận đấu.", "Đội A thua trận đấu."),
    ],
    "N": [
        ("Người đàn ông đang chạy.", "Người đàn ông đang chạy nhanh vì trễ giờ."),
        ("Cô ấy mua một chiếc váy.", "Cô ấy mua váy để đi dự tiệc tối nay."),
        ("Trời đang mưa.", "Ngày mai trời sẽ nắng."),
        ("Anh ấy sống ở Hà Nội.", "Anh ấy thích phở Hà Nội."),
        ("Cuốn sách nằm trên bàn.", "Cuốn sách rất thú vị."),
    ],
}

def generate_synthetic(num_train=3000, num_dev=500, num_test=800, seed=42, out_dir="data/raw"):
    random.seed(seed)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Slight class imbalance to mimic ViNLI
    weights = {"E": 0.34, "C": 0.28, "N": 0.38}
    for split, n in [("train", num_train), ("dev", num_dev), ("test", num_test)]:
        rows = []
        for idx in range(n):
            label = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            premise, hypothesis = random.choice(VI_TEMPLATES[label])
            # add small noise to make samples unique
            suffix = f" #{idx}" if random.random() < 0.05 else ""
            rows.append({"id": f"{split}-{idx:05d}", "premise": premise+suffix, "hypothesis": hypothesis, "label": label})
        # shuffle
        random.shuffle(rows)
        p = out / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[synthetic] {p} -> {n} samples | dist={pd.Series([r['label'] for r in rows]).value_counts().to_dict()}")
    return out

class NLIDataset(Dataset):
    """Returns tokenized tensors + labels for flat/hier."""
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256, head_design: str = "e_first"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if head_design not in HEAD_DESIGNS:
            raise ValueError(f"Unknown head_design {head_design!r}, expected one of {list(HEAD_DESIGNS)}")
        self.head_design = head_design
        self._gold_to_coarse = HEAD_DESIGNS[head_design]["gold_to_coarse"]
        self._gold_to_fine = HEAD_DESIGNS[head_design]["gold_to_fine"]
        self.samples: List[Dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"[NLIDataset] {jsonl_path}: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["premise"], s["hypothesis"],
            truncation=True, max_length=self.max_length, padding="max_length"
        )
        gold = s["label"]
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "gold_label": gold,
            "gold_id": LABEL2ID_FLAT[gold],
            "coarse_label": self._gold_to_coarse[gold],
            "fine_label": self._gold_to_fine[gold] if self._gold_to_fine[gold] is not None else -100,
            "sample_id": s["id"],
        }
        # token_type_ids if tokenizer provides
        if "token_type_ids" in enc:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        return item

def load_or_generate(config_path="configs/config.yaml"):
    """Ensure data exists; generate synthetic if missing."""
    import yaml
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    data_cfg = cfg["data"]
    raw_dir = pathlib.Path(data_cfg["raw_dir"])
    needed = [data_cfg["train_file"], data_cfg["dev_file"], data_cfg["test_file"]]
    if not all(pathlib.Path(p).exists() for p in needed):
        print("[data] raw files missing -> generating synthetic...")
        generate_synthetic(
            num_train=data_cfg["num_synthetic_train"],
            num_dev=data_cfg["num_synthetic_dev"],
            num_test=data_cfg["num_synthetic_test"],
            seed=cfg["project"]["seed"],
            out_dir=str(raw_dir),
        )
    else:
        print("[data] raw files found, skip generation")
    return cfg
