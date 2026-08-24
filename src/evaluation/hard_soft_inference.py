"""Offline Hard/Soft inference from saved hierarchical logits."""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None; F = None

LABELS = ["E","C","N"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

def logits_to_probs(logits):
    return F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

def compute_hard_soft(df: pd.DataFrame, head_design: str = None):
    """df must have columns matching head_design:
      e_first (Run A, default): coarse_logit_E, coarse_logit_nonE, fine_logit_C, fine_logit_N
      n_first (Run B):          coarse_logit_N, coarse_logit_nonN, fine_logit_E, fine_logit_C
    head_design=None -> auto-detect from which columns are present in df.
    Returns df with p_E, p_C, p_N, soft_pred, hard_pred (+ label strings)."""
    if head_design is None:
        head_design = "n_first" if "coarse_logit_N" in df.columns else "e_first"
    if head_design not in ("e_first", "n_first"):
        raise ValueError(f"Unknown head_design {head_design!r}, expected 'e_first' or 'n_first'")

    if head_design == "e_first":
        coarse = df[["coarse_logit_E", "coarse_logit_nonE"]].values.astype(np.float32)
        fine   = df[["fine_logit_C", "fine_logit_N"]].values.astype(np.float32)
    else:
        coarse = df[["coarse_logit_N", "coarse_logit_nonN"]].values.astype(np.float32)
        fine   = df[["fine_logit_E", "fine_logit_C"]].values.astype(np.float32)

    # softmax
    p_coarse = np.exp(coarse) / np.exp(coarse).sum(axis=1, keepdims=True)
    p_fine   = np.exp(fine)   / np.exp(fine).sum(axis=1, keepdims=True)

    if head_design == "e_first":
        p_E = p_coarse[:, 0]; p_non_root = p_coarse[:, 1]
        p_C = p_non_root * p_fine[:, 0]
        p_N = p_non_root * p_fine[:, 1]
    else:
        p_N = p_coarse[:, 0]; p_non_root = p_coarse[:, 1]
        p_E = p_non_root * p_fine[:, 0]
        p_C = p_non_root * p_fine[:, 1]

    p_soft = np.stack([p_E, p_C, p_N], axis=1)
    soft_pred = p_soft.argmax(axis=1)

    # hard
    coarse_pred = coarse.argmax(axis=1)  # 0:root 1:non-root
    fine_pred   = fine.argmax(axis=1)
    if head_design == "e_first":
        hard_pred = np.where(coarse_pred == 0, 0, np.where(fine_pred == 0, 1, 2))  # E / C / N
    else:
        hard_pred = np.where(coarse_pred == 0, 2, np.where(fine_pred == 0, 0, 1))  # N / E / C

    out = df.copy()
    out["p_E"] = p_E
    out["p_C"] = p_C
    out["p_N"] = p_N
    out["soft_pred"] = soft_pred
    out["hard_pred"] = hard_pred
    out["soft_pred_label"] = [ID2LABEL[i] for i in soft_pred]
    out["hard_pred_label"] = [ID2LABEL[i] for i in hard_pred]
    return out

def load_and_infer(hier_csv: str, out_hard_csv: str=None, out_soft_csv: str=None):
    df = pd.read_csv(hier_csv)
    df = compute_hard_soft(df)
    if out_hard_csv: df.to_csv(out_hard_csv, index=False)
    if out_soft_csv: df.to_csv(out_soft_csv, index=False)
    # also return combined
    return df
