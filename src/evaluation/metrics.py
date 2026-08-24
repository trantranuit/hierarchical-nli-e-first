"""Metrics, confusion, diagnostic utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

LABELS = ["E","C","N"]

def compute_metrics(y_true, y_pred, prefix=""):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])
    # ensure 3 values even if missing class
    if len(per_f1) < 3:
        tmp = [0,0,0]
        for i, v in enumerate(per_f1): tmp[i]=v
        per_f1 = np.array(tmp)
    return {
        f"{prefix}accuracy": acc,
        f"{prefix}macro_f1": macro_f1,
        f"{prefix}f1_E": per_f1[0],
        f"{prefix}f1_C": per_f1[1],
        f"{prefix}f1_N": per_f1[2],
    }

def confusion_df(y_true, y_pred, labels=[0,1,2], names=LABELS):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=[f"gold_{n}" for n in names], columns=[f"pred_{n}" for n in names])
    return df, cm

def coarse_metrics(y_true_flat, y_pred_coarse_or_flat):
    """Map flat gold/pred to coarse E vs Non-E and compute F1."""
    y_true_c = [0 if y==0 else 1 for y in y_true_flat]
    # y_pred may be flat (0,1,2) or coarse (0,1)
    if max(y_pred_coarse_or_flat) <= 1:
        y_pred_c = list(y_pred_coarse_or_flat)
    else:
        y_pred_c = [0 if y==0 else 1 for y in y_pred_coarse_or_flat]
    acc = accuracy_score(y_true_c, y_pred_c)
    f1 = f1_score(y_true_c, y_pred_c, average="macro", zero_division=0)
    return {"coarse_accuracy": acc, "coarse_macro_f1": f1}

def fine_metrics(y_true_flat, y_pred_flat, mask_nonE_gold=True):
    """F1 for C/N among Non-E gold samples."""
    if mask_nonE_gold:
        pairs = [(t,p) for t,p in zip(y_true_flat, y_pred_flat) if t in (1,2)]
        if not pairs: return {"fine_accuracy": 0, "fine_macro_f1": 0}
        yt, yp = zip(*pairs)
        # map C=0 N=1 for fine
        yt = [0 if y==1 else 1 for y in yt]
        yp = [0 if p==1 else (1 if p==2 else 0) for p in yp]  # if pred E when gold NonE -> count as C error
    else:
        yt = [0 if y==1 else 1 for y in y_true_flat if y in (1,2)]
        yp = [0 if p==1 else 1 for p in y_pred_flat]
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    return {"fine_accuracy": acc, "fine_macro_f1": f1}

def cn_confusion_rate(cm):
    """C↔N confusion rate: (C->N + N->C) / total C+N gold."""
    # cm rows gold E/C/N, cols pred E/C/N
    c_to_n = cm[1,2] if cm.shape[0]>2 else 0
    n_to_c = cm[2,1] if cm.shape[0]>2 else 0
    total_cn_gold = cm[1].sum() + cm[2].sum()
    return (c_to_n + n_to_c) / max(1, total_cn_gold)

def diagnostic_report(y_true, y_pred, save_path=None):
    metrics = compute_metrics(y_true, y_pred)
    df, cm = confusion_df(y_true, y_pred)
    metrics["C<->N_confusion_rate"] = cn_confusion_rate(cm)
    # pairwise errors
    e_c = cm[0,1] if cm.shape[0]>1 else 0
    e_n = cm[0,2] if cm.shape[0]>2 else 0
    c_e = cm[1,0] if cm.shape[0]>1 else 0
    n_e = cm[2,0] if cm.shape[0]>2 else 0
    metrics.update({"E->C": int(e_c), "E->N": int(e_n), "C->E": int(c_e), "N->E": int(n_e),
                    "C->N": int(cm[1,2]), "N->C": int(cm[2,1])})
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0)
    if save_path:
        df.to_csv(save_path.replace(".txt","_cm.csv"))
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report+"\n")
            f.write(str(df)+"\n")
            f.write(str(metrics)+"\n")
    return metrics, df, cm, report
