"""Compare Flat vs Hier-Hard vs Hier-Soft — final Step 5."""
from __future__ import annotations
import pathlib, json, pandas as pd
import numpy as np
from .metrics import compute_metrics, confusion_df, cn_confusion_rate, diagnostic_report

LABELS = ["E","C","N"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}

def evaluate_csv(csv_path, pred_col: str, gold_col: str="gold_label"):
    # csv_path may be str path or DataFrame
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path
    else:
        df = pd.read_csv(csv_path)
    def to_ids(series):
        # robust: handle string labels E/C/N or ints, regardless of pandas StringDtype
        vals = series.tolist()
        # if any element is string E/C/N -> map
        if len(vals)>0 and isinstance(vals[0], str):
            return [LABEL2ID[str(x)] if str(x) in LABEL2ID else int(x) for x in vals]
        # numpy/pandas may give StringDtype that looks object-like, check via try
        try:
            return [int(x) for x in vals]
        except Exception:
            return [LABEL2ID[str(x)] if str(x) in LABEL2ID else int(float(x)) for x in vals]
    y_true = to_ids(df[gold_col])
    y_pred = to_ids(df[pred_col])
    metrics, cm_df, cm, report = diagnostic_report(y_true, y_pred)
    return metrics, cm_df, cm, report, df

def compare_all(flat_csv: str, hier_csv: str, out_dir: str="outputs/comparison"):
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from .hard_soft_inference import compute_hard_soft
    hier_df = pd.read_csv(hier_csv)
    hier_df = compute_hard_soft(hier_df)
    flat_metrics, flat_cm_df, flat_cm, flat_report, flat_df = evaluate_csv(flat_csv, "pred_flat")
    hard_metrics, hard_cm_df, hard_cm, hard_report, _ = evaluate_csv(hier_df, "hard_pred")
    soft_metrics, soft_cm_df, soft_cm, soft_report, _ = evaluate_csv(hier_df, "soft_pred")

    # Build comparison table
    rows = []
    for name, m in [("Flat", flat_metrics), ("Hier-Hard", hard_metrics), ("Hier-Soft", soft_metrics)]:
        rows.append({"Model": name, **m})
    comp = pd.DataFrame(rows)
    comp_path = out_dir / "comparison_metrics.csv"
    comp.to_csv(comp_path, index=False)

    # Save CMs
    flat_cm_df.to_csv(out_dir / "flat_cm.csv")
    hard_cm_df.to_csv(out_dir / "hard_cm.csv")
    soft_cm_df.to_csv(out_dir / "soft_cm.csv")

    # Detailed report
    report_path = out_dir / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# So sánh cuối — Flat vs Hier-Hard vs Hier-Soft\n\n")
        f.write(comp.to_markdown(index=False))
        f.write("\n\n## Flat — classification_report\n```\n"+flat_report+"\n```\n")
        f.write("\n## Hier-Hard — classification_report\n```\n"+hard_report+"\n```\n")
        f.write("\n## Hier-Soft — classification_report\n```\n"+soft_report+"\n```\n")
        f.write("\n## Confusion Matrices\n\n### Flat\n"+flat_cm_df.to_markdown()+"\n\n### Hier-Hard\n"+hard_cm_df.to_markdown()+"\n\n### Hier-Soft\n"+soft_cm_df.to_markdown()+"\n")
        # GO/STOP
        flat_macro = flat_metrics["macro_f1"]
        hard_macro = hard_metrics["macro_f1"]
        soft_macro = soft_metrics["macro_f1"]
        best_hier = max(hard_macro, soft_macro)
        f.write(f"\n## GO / STOP\n\n- Flat Macro-F1: {flat_macro:.4f}\n- Hier-Hard Macro-F1: {hard_macro:.4f}\n- Hier-Soft Macro-F1: {soft_macro:.4f}\n")
        f.write(f"- Flat C↔N confusion: {flat_metrics['C<->N_confusion_rate']:.4f}\n")
        f.write(f"- Hier-Hard C↔N: {hard_metrics['C<->N_confusion_rate']:.4f}\n")
        f.write(f"- Hier-Soft C↔N: {soft_metrics['C<->N_confusion_rate']:.4f}\n")
        if best_hier > flat_macro + 0.005:
            f.write("\n**GO** — Hierarchical tốt hơn Flat (Macro-F1).\n")
        elif best_hier < flat_macro - 0.01:
            f.write("\n**STOP / reconsider** — Hierarchical kém Flat rõ ràng.\n")
        else:
            f.write("\n**NEUTRAL** — Chênh lệch nhỏ, cần xem F1 C/N và Head1 E/Non-E.\n")
    print(f"[compare] saved -> {comp_path} | {report_path}")
    print(comp.to_string(index=False))
    return comp, report_path
