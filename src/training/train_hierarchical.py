"""Train Hierarchical CafeBERT 2-head — Run 2.

Supports 2 head_design (configs/config.yaml -> model.head_design), song song, không loại trừ nhau:
  e_first (Run A, mặc định): Head1(E/Non-E) + Head2(C/N)  — cột coarse_logit_E/nonE, fine_logit_C/N
  n_first (Run B):           Head1(N/Non-N) + Head2(E/C)  — cột coarse_logit_N/nonN, fine_logit_E/C

Tracking: W&B only (hyperparams, seed, lr, batch_size, train/dev loss, Accuracy,
Macro-F1, F1 per class incl. coarse/fine heads, confusion matrix, run metadata).
Mandatory storage: best checkpoint + tokenizer/config + dev/test per-sample logits
are pushed to a PRIVATE Hugging Face Hub repo (configs/config.yaml -> hf_hub.hier_repo_id).
"""
from __future__ import annotations
import argparse, pathlib, yaml, json, random
import numpy as np, torch, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from src.data.dataset import NLIDataset, LABEL2ID_FLAT, load_or_generate
from src.models.hierarchical_cafebert import HierarchicalCafeBERT
from src.evaluation.metrics import compute_metrics, diagnostic_report
from src.evaluation.hard_soft_inference import compute_hard_soft
from src.utils import wandb_helper, hf_hub_helper

LABEL2ID = LABEL2ID_FLAT


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_tok(name, fallbacks):
    for n in [name] + fallbacks:
        try:
            t = AutoTokenizer.from_pretrained(n, trust_remote_code=True)
            print(f"[tokenizer] {n} ✓"); return t
        except Exception as e:
            print(f"[tokenizer] {n} fail {e}")
    raise RuntimeError("tokenizer fail")


def train(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    set_seed(cfg["project"]["seed"])
    load_or_generate(args.config)

    mcfg = cfg["model"]; dcfg = cfg["data"]; trcfg = cfg["training"]["hier"]; outcfg = cfg["outputs"]
    head_design = mcfg.get("head_design", "e_first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_hier] device={device} lambda={trcfg['lambda_fine']} head_design={head_design}")

    tok = get_tok(mcfg["cafebert_name"], mcfg["fallback_models"])
    train_ds = NLIDataset(dcfg["train_file"], tok, mcfg["max_length"], head_design=head_design)
    dev_ds   = NLIDataset(dcfg["dev_file"], tok, mcfg["max_length"], head_design=head_design)
    test_ds  = NLIDataset(dcfg["test_file"], tok, mcfg["max_length"], head_design=head_design) if pathlib.Path(dcfg["test_file"]).exists() else None
    if args.debug:
        for ds in [train_ds, dev_ds, test_ds]:
            if ds: ds.samples = ds.samples[:200]

    # --- W&B run (tracking only) ---
    run = None
    if not args.mock:
        hparams = {
            "model": mcfg["cafebert_name"], "max_length": mcfg["max_length"], "dropout": mcfg["dropout"],
            "epochs": trcfg["epochs"], "early_stopping_patience_evals": trcfg.get("early_stopping_patience"),
            "eval_steps": trcfg.get("eval_steps"),
            "lr": trcfg["lr"], "batch_size": trcfg["batch_size"],
            "grad_accum_steps": trcfg.get("grad_accum_steps", 1),
            "effective_batch_size": trcfg["batch_size"] * trcfg.get("grad_accum_steps", 1),
            "fp16": trcfg.get("fp16", False), "gradient_checkpointing": mcfg.get("gradient_checkpointing", False),
            "label_smoothing": mcfg.get("label_smoothing", 0.0), "head_design": head_design,
            "warmup_ratio": trcfg["warmup_ratio"], "weight_decay": trcfg["weight_decay"],
            "lambda_fine": trcfg["lambda_fine"], "seed": cfg["project"]["seed"],
            "num_train": len(train_ds), "num_dev": len(dev_ds), "num_test": len(test_ds) if test_ds else 0,
        }
        run = wandb_helper.init_run(cfg, "hier", hparams)

    model = HierarchicalCafeBERT(mcfg["cafebert_name"], mcfg["fallback_models"], mcfg["dropout"], lambda_fine=trcfg["lambda_fine"],
                                 gradient_checkpointing=mcfg.get("gradient_checkpointing", False),
                                 label_smoothing=mcfg.get("label_smoothing", 0.0), head_design=head_design)
    model.to(device)

    grad_accum = max(1, trcfg.get("grad_accum_steps", 1))
    fp16_enabled = device.type == "cuda" and trcfg.get("fp16", False)
    scaler = torch.amp.GradScaler(device.type, enabled=fp16_enabled)
    print(f"[train_hier] grad_accum_steps={grad_accum} fp16={fp16_enabled} effective_batch={trcfg['batch_size']*grad_accum}")

    def collate(batch):
        keys = ["input_ids", "attention_mask"]
        if "token_type_ids" in batch[0]: keys.append("token_type_ids")
        out = {k: torch.stack([b[k] for b in batch]) for k in keys}
        out["coarse_labels"] = torch.tensor([b["coarse_label"] for b in batch], dtype=torch.long)
        out["fine_labels"] = torch.tensor([b["fine_label"] for b in batch], dtype=torch.long)
        out["gold_ids"] = torch.tensor([b["gold_id"] for b in batch], dtype=torch.long)
        out["sample_id"] = [b["sample_id"] for b in batch]
        out["gold_label"] = [b["gold_label"] for b in batch]
        return out

    num_workers = trcfg.get("num_workers", 0)
    dl_kwargs = dict(num_workers=num_workers, pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=trcfg["batch_size"], shuffle=True, collate_fn=collate, **dl_kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=trcfg["batch_size"] * 2, collate_fn=collate, **dl_kwargs)

    best_path = pathlib.Path(outcfg["hier_ckpt"]) / "best"; best_path.mkdir(parents=True, exist_ok=True)

    if args.mock:
        print("[train_hier] MOCK mode — random logits, skip training (no W&B/HF push)")
        return mock_predict(dev_ds, outcfg, dcfg, head_design=head_design)

    optimizer = torch.optim.AdamW(model.parameters(), lr=trcfg["lr"], weight_decay=trcfg["weight_decay"])
    steps_per_epoch = -(-len(train_loader) // grad_accum)  # ceil: 1 optimizer step per grad_accum batches
    total = steps_per_epoch * trcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * trcfg["warmup_ratio"]), total)
    best = -1
    patience = trcfg.get("early_stopping_patience")  # in EVALS now, not epochs (eval runs every eval_steps)
    eval_steps = trcfg.get("eval_steps")
    evals_no_improve = 0
    stop_training = False

    global_step = 0
    for epoch in range(1, trcfg["epochs"] + 1):
        if stop_training:
            break
        model.train(); tot = 0
        optimizer.zero_grad()
        n_batches = len(train_loader)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Hier Epoch {epoch}/{trcfg['epochs']}"), start=1):
            tens = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids", "coarse_labels", "fine_labels")}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16_enabled):
                out = model(**tens)
                loss = out["loss"] / grad_accum
            scaler.scale(loss).backward()
            tot += loss.item() * grad_accum

            is_last_batch = batch_idx == n_batches
            if batch_idx % grad_accum == 0 or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= scale_before:
                    # only step the LR schedule if the optimizer step actually happened —
                    # GradScaler skips it on inf/nan grads (e.g. while scale auto-calibrates)
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % trcfg["logging_steps"] == 0:
                    wandb_helper.log_step(run, {
                        "train/loss": tot / batch_idx,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch_progress": epoch - 1 + batch_idx / n_batches,
                    }, step=global_step)

                # eval + save-best at eval_steps granularity (not just once/epoch) — see train_flat.py
                # for the rationale. Always also eval at the last batch of an epoch as a safety net.
                should_eval = (eval_steps and global_step % eval_steps == 0) or is_last_batch
                if should_eval:
                    dev_metrics, dev_loss = evaluate(model, dev_loader, device, compute_loss=True, head_design=head_design)
                    print(f"[eval step {global_step} epoch {epoch}] {dev_metrics} dev_loss={dev_loss:.4f}")
                    dev_payload = {f"dev/{k}": v for k, v in dev_metrics.items()}
                    dev_payload["dev/loss"] = dev_loss
                    dev_payload["epoch"] = epoch
                    wandb_helper.log_step(run, dev_payload, step=global_step)

                    cur = dev_metrics["soft_macro_f1"]  # primary selection metric
                    if cur > best:
                        best = cur
                        evals_no_improve = 0
                        torch.save(model.state_dict(), best_path / "pytorch_model.bin")
                        tok.save_pretrained(best_path)
                        print(f"  ★ best soft_macro {best:.4f} (step {global_step})")
                    else:
                        evals_no_improve += 1
                        print(f"  no improvement ({evals_no_improve}/{patience or '∞'} eval(s))")
                        if patience is not None and evals_no_improve >= patience:
                            print(f"[train_hier] early stopping — dev soft_macro_f1 không cải thiện sau {patience} eval liên tiếp (best={best:.4f})")
                            wandb_helper.log_step(run, {"train/early_stopped_at_step": global_step})
                            stop_training = True
                    model.train()
                    if stop_training:
                        break
        train_loss_epoch = tot / len(train_loader)
        print(f"[epoch {epoch}] loss={train_loss_epoch:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        wandb_helper.log_epoch_metrics(run, "train", {"loss": train_loss_epoch}, epoch)

    try:
        model.load_state_dict(torch.load(best_path / "pytorch_model.bin", map_location=device))
    except Exception as e:
        print(f"[train_hier] could not reload best checkpoint: {e}")

    pred_paths = {}
    final_metrics = {}
    for split, ds, loader in [("dev", dev_ds, dev_loader)] + ([("test", test_ds, torch.utils.data.DataLoader(test_ds, batch_size=trcfg["batch_size"] * 2, collate_fn=collate, **dl_kwargs))] if test_ds else []):
        df = evaluate(model, loader, device, return_df=True, head_design=head_design)
        out_csv = pathlib.Path(outcfg["hier_pred"]) / f"{split}_predictions.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        pred_paths[split] = out_csv
        print(f"[predict {split}] -> {out_csv} ({len(df)} rows)")

        df2 = compute_hard_soft(df, head_design=head_design)
        df2.to_csv(pathlib.Path(outcfg["hier_pred"]) / f"{split}_predictions_with_hard_soft.csv", index=False)

        split_metrics = {}
        for pred_col in ["hard_pred", "soft_pred"]:
            y_true = [LABEL2ID[g] for g in df2["gold_label"]]
            y_pred = df2[pred_col].tolist()
            mets, cm_df, cm, rep = diagnostic_report(y_true, y_pred)
            met_path = pathlib.Path(outcfg["metrics"]) / f"hier_{pred_col}_{split}_metrics.json"
            met_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(mets, open(met_path, "w"), indent=2)
            split_metrics[pred_col] = mets
            wandb_helper.log_epoch_metrics(run, f"final_{split}_{pred_col}", mets, trcfg["epochs"])
            wandb_helper.log_confusion_matrix(run, y_true, y_pred, ["E", "C", "N"], f"confusion_matrix/{split}_{pred_col}")

        # coarse/fine head metrics
        root_label = "E" if head_design == "e_first" else "N"
        coarse_col, coarse_nonroot_col = (("coarse_logit_E", "coarse_logit_nonE") if head_design == "e_first"
                                          else ("coarse_logit_N", "coarse_logit_nonN"))
        y_true_c = [0 if g == root_label else 1 for g in df["gold_label"]]
        y_pred_c = (df[coarse_nonroot_col] > df[coarse_col]).astype(int).tolist()
        coarse_f1 = f1_score(y_true_c, y_pred_c, average="macro", zero_division=0)
        print(f"  Head1 ({root_label}/Non-{root_label}) coarse macro-F1: {coarse_f1:.4f}")
        split_metrics["coarse_macro_f1"] = coarse_f1
        wandb_helper.log_step(run, {f"final_{split}/coarse_macro_f1": coarse_f1})

        final_metrics[split] = split_metrics

    # --- MANDATORY storage: push checkpoint + tokenizer + predictions to private HF Hub repo ---
    run_metadata = {
        "run_type": "hier", "model": model.model_name_used if hasattr(model, "model_name_used") else mcfg["cafebert_name"],
        "seed": cfg["project"]["seed"], "hparams": trcfg, "metrics": final_metrics,
        "wandb_run_url": run.url if run else None,
    }
    repo_id, revision = hf_hub_helper.push_run_artifacts(cfg, "hier", best_path, pred_paths, run_metadata)
    if repo_id:
        wandb_helper.log_hf_link(run, repo_id, revision)

    wandb_helper.finish(run)
    return final_metrics


def evaluate(model, loader, device, return_df=False, compute_loss=False, head_design="e_first"):
    coarse_col, coarse_nonroot_col = (("coarse_logit_E", "coarse_logit_nonE") if head_design == "e_first"
                                      else ("coarse_logit_N", "coarse_logit_nonN"))
    fine_col0, fine_col1 = (("fine_logit_C", "fine_logit_N") if head_design == "e_first"
                            else ("fine_logit_E", "fine_logit_C"))
    model.eval()
    coarse_logits = []; fine_logits = []; all_ids = []; all_gold = []
    total_loss = 0.0; n_batches = 0
    with torch.no_grad():
        for batch in loader:
            tens = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            labels = None
            if compute_loss:
                labels = {"coarse_labels": batch["coarse_labels"].to(device), "fine_labels": batch["fine_labels"].to(device)}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(**tens, **(labels or {}))
            if compute_loss and out["loss"] is not None:
                total_loss += out["loss"].item(); n_batches += 1
            coarse_logits.append(out["coarse_logits"].cpu().numpy())
            fine_logits.append(out["fine_logits"].cpu().numpy())
            all_ids.extend(batch["sample_id"]); all_gold.extend(batch["gold_label"])
    coarse_logits = np.concatenate(coarse_logits, axis=0)
    fine_logits = np.concatenate(fine_logits, axis=0)
    avg_loss = total_loss / n_batches if n_batches else None
    if return_df:
        df = pd.DataFrame({"sample_id": all_ids, "gold_label": all_gold,
                           coarse_col: coarse_logits[:, 0], coarse_nonroot_col: coarse_logits[:, 1],
                           fine_col0: fine_logits[:, 0], fine_col1: fine_logits[:, 1]})
        return df
    df = pd.DataFrame({coarse_col: coarse_logits[:, 0], coarse_nonroot_col: coarse_logits[:, 1],
                       fine_col0: fine_logits[:, 0], fine_col1: fine_logits[:, 1], "gold_label": all_gold})
    df2 = compute_hard_soft(df, head_design=head_design)
    y_true = [LABEL2ID[g] for g in all_gold]
    hard = compute_metrics(y_true, df2["hard_pred"].tolist()) if y_true else {}
    soft = compute_metrics(y_true, df2["soft_pred"].tolist())
    metrics = {**{f"soft_{k}": v for k, v in soft.items()}, **{f"hard_{k}": v for k, v in hard.items()}}
    if compute_loss:
        return metrics, avg_loss
    return metrics, None


def mock_predict(dev_ds, outcfg, dcfg, head_design="e_first"):
    """Local-only debug aid to verify the pipeline shape — never pushed to W&B/HF."""
    np.random.seed(123)
    root_label = "E" if head_design == "e_first" else "N"
    leaf_labels = ("C", "N") if head_design == "e_first" else ("E", "C")
    coarse_col, coarse_nonroot_col = (("coarse_logit_E", "coarse_logit_nonE") if head_design == "e_first"
                                      else ("coarse_logit_N", "coarse_logit_nonN"))
    fine_col0, fine_col1 = (("fine_logit_C", "fine_logit_N") if head_design == "e_first"
                            else ("fine_logit_E", "fine_logit_C"))
    for split in ["dev", "test"]:
        src = dcfg[f"{split}_file"]
        if not pathlib.Path(src).exists(): continue
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        data = []
        for r in rows:
            gold = r["label"]
            cl = np.random.randn(2); fl = np.random.randn(2)
            if gold == root_label: cl[0] += 1.0
            else: cl[1] += 1.0
            if gold == leaf_labels[0]: fl[0] += 1.0
            elif gold == leaf_labels[1]: fl[1] += 1.0
            if np.random.rand() < 0.15:
                cl = np.flip(cl)
            data.append({"sample_id": r["id"], "gold_label": gold,
                        coarse_col: float(cl[0]), coarse_nonroot_col: float(cl[1]),
                        fine_col0: float(fl[0]), fine_col1: float(fl[1])})
        df = pd.DataFrame(data)
        out = pathlib.Path(outcfg["hier_pred"]) / f"{split}_predictions.csv"; out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        df2 = compute_hard_soft(df, head_design=head_design)
        df2.to_csv(pathlib.Path(outcfg["hier_pred"]) / f"{split}_predictions_with_hard_soft.csv", index=False)
        print(f"[mock hier] {out} ({len(df)} rows) -> also with_hard_soft")
        from src.evaluation.metrics import diagnostic_report
        for col in ["hard_pred", "soft_pred"]:
            y_true = [LABEL2ID[g] for g in df2["gold_label"]]; y_pred = df2[col].tolist()
            mets, cm_df, cm, rep = diagnostic_report(y_true, y_pred)
            met_path = pathlib.Path(outcfg["metrics"]) / f"hier_{col}_{split}_metrics.json"; met_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(mets, open(met_path, "w"), indent=2)
    print("[mock hier] done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mock", action="store_true", help="local only, no W&B/HF")
    args = ap.parse_args()
    train(args)
