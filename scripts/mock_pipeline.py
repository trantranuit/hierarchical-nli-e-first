#!/usr/bin/env python3
"""Standalone mock pipeline — no torch/transformers needed. Generates all outputs."""
import json, random, pathlib, sys
import pandas as pd, numpy as np, yaml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/config.yaml"
cfg = yaml.safe_load(open(CFG_PATH, encoding="utf-8"))

LABELS=["E","C","N"]
L2I={l:i for i,l in enumerate(LABELS)}
I2L={i:l for l,i in L2I.items()}

# Import synthetic gen without torch
sys.path.insert(0, str(ROOT))
from src.data.dataset import generate_synthetic

# 0. data
raw_dir = ROOT / cfg["data"]["raw_dir"]
needed=[ROOT/cfg["data"]["train_file"], ROOT/cfg["data"]["dev_file"], ROOT/cfg["data"]["test_file"]]
if not all(p.exists() for p in needed):
    generate_synthetic(cfg["data"]["num_synthetic_train"], cfg["data"]["num_synthetic_dev"], cfg["data"]["num_synthetic_test"], cfg["project"]["seed"], str(ROOT/cfg["data"]["raw_dir"]))

def metrics_and_cm(y_true, y_pred):
    acc=accuracy_score(y_true,y_pred)
    macro=f1_score(y_true,y_pred,average="macro",zero_division=0)
    per=f1_score(y_true,y_pred,average=None,labels=[0,1,2],zero_division=0)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1,2])
    df=pd.DataFrame(cm,index=[f"gold_{n}" for n in LABELS],columns=[f"pred_{n}" for n in LABELS])
    c2n=(cm[1,2]+cm[2,1])/max(1,cm[1].sum()+cm[2].sum())
    rep=classification_report(y_true,y_pred,target_names=LABELS,digits=4,zero_division=0)
    mets={"accuracy":acc,"macro_f1":macro,"f1_E":per[0],"f1_C":per[1],"f1_N":per[2],"C<->N_confusion_rate":c2n,
           "E->C":int(cm[0,1]),"E->N":int(cm[0,2]),"C->E":int(cm[1,0]),"N->E":int(cm[2,0]),"C->N":int(cm[1,2]),"N->C":int(cm[2,1])}
    return mets,df,cm,rep

# 1. Flat mock predictions
for split in ["dev","test"]:
    src=ROOT/f"data/raw/{split}.jsonl"
    if not src.exists(): continue
    rows=[json.loads(l) for l in open(src,encoding="utf-8")]
    np.random.seed(42 if split=="dev" else 123)
    data=[]
    for r in rows:
        gold=r["label"]; gid=L2I[gold]
        logits=np.random.randn(3)
        if np.random.rand()<0.64: logits[gid]+=1.3  # ~64% correct
        pred=int(logits.argmax())
        data.append({"sample_id":r["id"],"gold_label":gold,"logit_E":float(logits[0]),"logit_C":float(logits[1]),"logit_N":float(logits[2]),"pred_flat":pred,"pred_label":I2L[pred]})
    df=pd.DataFrame(data)
    out=ROOT/f"outputs/predictions/flat/{split}_predictions.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    y_true=[L2I[g] for g in df["gold_label"]]; y_pred=df["pred_flat"].tolist()
    mets,cm_df,cm,rep=metrics_and_cm(y_true,y_pred)
    (ROOT/f"outputs/metrics").mkdir(parents=True,exist_ok=True)
    json.dump(mets,open(ROOT/f"outputs/metrics/flat_{split}_metrics.json","w"),indent=2)
    (ROOT/f"outputs/diagnostics").mkdir(parents=True,exist_ok=True)
    cm_df.to_csv(ROOT/f"outputs/diagnostics/flat_{split}_cm.csv")
    open(ROOT/f"outputs/diagnostics/flat_{split}_report.txt","w").write(rep)
    print(f"[flat {split}] {out} acc={mets['accuracy']:.3f} macro={mets['macro_f1']:.3f} C↔N={mets['C<->N_confusion_rate']:.3f}")

# 3. Hierarchical mock predictions + 4A/4B
for split in ["dev","test"]:
    src=ROOT/f"data/raw/{split}.jsonl"
    if not src.exists(): continue
    rows=[json.loads(l) for l in open(src,encoding="utf-8")]
    np.random.seed(202 if split=="dev" else 303)
    data=[]
    for r in rows:
        gold=r["label"]
        cl=np.random.randn(2); fl=np.random.randn(2)
        # Head1: E vs Non-E
        if gold=="E": cl[0]+=1.2
        else: cl[1]+=1.2
        # Head2: C vs N (only for Non-E, but generate anyway)
        if gold=="C": fl[0]+=1.0
        elif gold=="N": fl[1]+=1.0
        else: fl+=np.random.randn(2)*0.3
        # inject Head1 errors ~14%
        if np.random.rand()<0.14: cl=np.flip(cl)
        # inject Head2 errors ~18% when Non-E
        if gold in ("C","N") and np.random.rand()<0.10: fl=np.flip(fl)
        data.append({"sample_id":r["id"],"gold_label":gold,"coarse_logit_E":float(cl[0]),"coarse_logit_nonE":float(cl[1]),"fine_logit_C":float(fl[0]),"fine_logit_N":float(fl[1])})
    df=pd.DataFrame(data)
    out=ROOT/f"outputs/predictions/hier/{split}_predictions.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    # hard/soft
    coarse=df[["coarse_logit_E","coarse_logit_nonE"]].values.astype(np.float32)
    fine=df[["fine_logit_C","fine_logit_N"]].values.astype(np.float32)
    p_coarse=np.exp(coarse)/np.exp(coarse).sum(axis=1,keepdims=True)
    p_fine=np.exp(fine)/np.exp(fine).sum(axis=1,keepdims=True)
    p_E=p_coarse[:,0]; p_NonE=p_coarse[:,1]; p_C=p_NonE*p_fine[:,0]; p_N=p_NonE*p_fine[:,1]
    p_soft=np.stack([p_E,p_C,p_N],axis=1); soft_pred=p_soft.argmax(axis=1)
    coarse_pred=coarse.argmax(axis=1); fine_pred=fine.argmax(axis=1)
    hard_pred=np.where(coarse_pred==0,0,np.where(fine_pred==0,1,2))
    df2=df.copy(); df2["p_E"]=p_E; df2["p_C"]=p_C; df2["p_N"]=p_N; df2["soft_pred"]=soft_pred; df2["hard_pred"]=hard_pred
    df2["soft_pred_label"]=[I2L[i] for i in soft_pred]; df2["hard_pred_label"]=[I2L[i] for i in hard_pred]
    df2.to_csv(ROOT/f"outputs/predictions/hier/{split}_predictions_with_hard_soft.csv",index=False)
    for col in ["hard_pred","soft_pred"]:
        y_true=[L2I[g] for g in df2["gold_label"]]; y_pred=df2[col].tolist()
        mets,cm_df,cm,rep=metrics_and_cm(y_true,y_pred)
        json.dump(mets,open(ROOT/f"outputs/metrics/hier_{col}_{split}_metrics.json","w"),indent=2)
        cm_df.to_csv(ROOT/f"outputs/diagnostics/hier_{col}_{split}_cm.csv")
        open(ROOT/f"outputs/diagnostics/hier_{col}_{split}_report.txt","w").write(rep)
        print(f"[hier {col} {split}] macro={mets['macro_f1']:.3f} C↔N={mets['C<->N_confusion_rate']:.3f}")
    # coarse head stats
    y_true_c=[0 if g=="E" else 1 for g in df["gold_label"]]
    y_pred_c=(df["coarse_logit_nonE"]>df["coarse_logit_E"]).astype(int).tolist()
    coarse_acc=accuracy_score(y_true_c,y_pred_c); coarse_f1=f1_score(y_true_c,y_pred_c,average="macro",zero_division=0)
    print(f"  Head1 E/Non-E {split}: acc={coarse_acc:.3f} macroF1={coarse_f1:.3f}")

# 5. Comparison
from src.evaluation.compare import compare_all
flat_p=ROOT/"outputs/predictions/flat/test_predictions.csv"
hier_p=ROOT/"outputs/predictions/hier/test_predictions.csv"
if not flat_p.exists(): flat_p=ROOT/"outputs/predictions/flat/dev_predictions.csv"
if not hier_p.exists(): hier_p=ROOT/"outputs/predictions/hier/dev_predictions.csv"
comp, rp = compare_all(str(flat_p), str(hier_p), out_dir=str(ROOT/"outputs/comparison"))
print("\n=== COMPARISON ===")
print(comp.to_string(index=False))
print(f"\nReport -> {rp}")
# also copy raw data manifest
manifest={"project":cfg["project"],"model":cfg["model"],"training":cfg["training"],"note":"mock pipeline — replace with real training for publication numbers; all CSV schemas per spec"}
json.dump(manifest,open(ROOT/"outputs/comparison/manifest.json","w"),indent=2,ensure_ascii=False)
print("\n[mock_pipeline] DONE — all outputs in one folder ✓")
