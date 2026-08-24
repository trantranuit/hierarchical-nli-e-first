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

def compute_hard_soft(df: pd.DataFrame):
    """df must have columns: coarse_logit_E, coarse_logit_nonE, fine_logit_C, fine_logit_N
    Returns df with p_E, p_C, p_N, soft_pred, hard_pred (+ label strings)."""
    coarse = df[["coarse_logit_E","coarse_logit_nonE"]].values.astype(np.float32)
    fine   = df[["fine_logit_C","fine_logit_N"]].values.astype(np.float32)
    # softmax
    p_coarse = np.exp(coarse) / np.exp(coarse).sum(axis=1, keepdims=True)
    p_fine   = np.exp(fine)   / np.exp(fine).sum(axis=1, keepdims=True)
    p_E = p_coarse[:,0]
    p_NonE = p_coarse[:,1]
    p_C_given = p_fine[:,0]
    p_N_given = p_fine[:,1]
    p_C = p_NonE * p_C_given
    p_N = p_NonE * p_N_given
    p_soft = np.stack([p_E, p_C, p_N], axis=1)
    soft_pred = p_soft.argmax(axis=1)
    # hard
    coarse_pred = coarse.argmax(axis=1)  # 0:E 1:NonE
    fine_pred   = fine.argmax(axis=1)    # 0:C 1:N
    hard_pred = np.where(coarse_pred==0, 0, np.where(fine_pred==0, 1, 2))

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
