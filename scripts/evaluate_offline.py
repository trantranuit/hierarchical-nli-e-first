"""Offline pipeline: diagnostic flat + hard/soft + comparison — no training."""
import argparse, pathlib, sys, yaml, json, pandas as pd

# Run as a plain script (`python3 scripts/evaluate_offline.py`) — repo root isn't on
# sys.path by default (only scripts/ is), so `from src....` would fail. Bootstrap it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.evaluation.hard_soft_inference import compute_hard_soft
from src.evaluation.metrics import diagnostic_report
from src.evaluation.compare import compare_all

LABELS=["E","C","N"]; L2I={l:i for i,l in enumerate(LABELS)}

def main(args):
    cfg=yaml.safe_load(open(args.config, encoding="utf-8"))
    out=cfg["outputs"]
    flat_dev = pathlib.Path(out["flat_pred"])/"dev_predictions.csv"
    hier_dev = pathlib.Path(out["hier_pred"])/"dev_predictions.csv"
    flat_test = pathlib.Path(out["flat_pred"])/"test_predictions.csv"
    hier_test = pathlib.Path(out["hier_pred"])/"test_predictions.csv"

    # Step 2 diagnostic flat
    for split, p in [("dev",flat_dev),("test",flat_test)]:
        if not p.exists(): print(f"[skip] {p} missing"); continue
        df=pd.read_csv(p)
        y_true=[L2I[g] for g in df["gold_label"]]; y_pred=df["pred_flat"].tolist()
        mets, cm_df, cm, rep = diagnostic_report(y_true, y_pred)
        diag_dir=pathlib.Path(out["diagnostics"]); diag_dir.mkdir(parents=True, exist_ok=True)
        cm_df.to_csv(diag_dir / f"flat_{split}_cm.csv")
        open(diag_dir / f"flat_{split}_report.txt","w").write(rep)
        print(f"[Step2 diagnostic flat {split}] {mets}")

    # Steps 4A/4B: enrich hier with hard/soft
    for split, p in [("dev",hier_dev),("test",hier_test)]:
        if not p.exists(): print(f"[skip] {p} missing"); continue
        df=pd.read_csv(p)
        df2=compute_hard_soft(df)
        out_path=pathlib.Path(out["hier_pred"])/f"{split}_predictions_with_hard_soft.csv"
        df2.to_csv(out_path, index=False)
        print(f"[Step4 hard/soft {split}] -> {out_path}")
        for col in ["hard_pred","soft_pred"]:
            y_true=[L2I[g] for g in df2["gold_label"]]; y_pred=df2[col].tolist()
            mets, cm_df, cm, rep = diagnostic_report(y_true, y_pred)
            json.dump(mets, open(pathlib.Path(out["metrics"])/f"hier_{col}_{split}_metrics.json","w"), indent=2)
            diag_dir=pathlib.Path(out["diagnostics"])
            cm_df.to_csv(diag_dir / f"hier_{col}_{split}_cm.csv")
            open(diag_dir / f"hier_{col}_{split}_report.txt","w").write(rep)
            print(f"  {col} {split}: {mets}")

    # Step 5 comparison (prefer test if exists else dev)
    flat_p = flat_test if flat_test.exists() else flat_dev
    hier_p = hier_test if hier_test.exists() else hier_dev
    if flat_p.exists() and hier_p.exists():
        # need enriched hier; if only base exists, compare will enrich internally
        comp, rp = compare_all(str(flat_p), str(hier_p), out_dir=out["comparison"])
        print(f"[Step5 comparison] -> {rp}")
        print(comp.to_string(index=False))
    else:
        print("[Step5] not enough predictions to compare")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args=ap.parse_args()
    main(args)
