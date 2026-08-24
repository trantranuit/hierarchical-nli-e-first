"""Train Flat CafeBERT — Run 1 (2 training runs total).

Tracking: W&B only (hyperparams, seed, lr, batch_size, train/dev loss, Accuracy,
Macro-F1, F1 per class, confusion matrix, run metadata).
Mandatory storage: best checkpoint + tokenizer/config + dev/test per-sample logits
are pushed to a PRIVATE Hugging Face Hub repo (configs/config.yaml -> hf_hub.flat_repo_id).
"""
from __future__ import annotations
import argparse, pathlib, yaml, json, random
import numpy as np, torch, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from src.data.dataset import NLIDataset, LABEL2ID_FLAT, load_or_generate
from src.models.flat_cafebert import FlatCafeBERT
from src.evaluation.metrics import diagnostic_report, compute_metrics
from src.utils import wandb_helper, hf_hub_helper


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_tokenizer(model_name, fallbacks):
    for name in [model_name] + fallbacks:
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            print(f"[tokenizer] loaded {name}")
            return tok
        except Exception as e:
            print(f"[tokenizer] {name} fail: {e}")
    raise RuntimeError("tokenizer failed")


def train(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    set_seed(cfg["project"]["seed"])
    load_or_generate(args.config)

    model_cfg = cfg["model"]; data_cfg = cfg["data"]; tr_cfg = cfg["training"]["flat"]; out_cfg = cfg["outputs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_flat] device={device}")

    tok = get_tokenizer(model_cfg["cafebert_name"], model_cfg["fallback_models"])
    train_ds = NLIDataset(data_cfg["train_file"], tok, model_cfg["max_length"])
    dev_ds   = NLIDataset(data_cfg["dev_file"], tok, model_cfg["max_length"])
    test_ds  = NLIDataset(data_cfg["test_file"], tok, model_cfg["max_length"]) if pathlib.Path(data_cfg["test_file"]).exists() else None

    if args.debug:
        for ds in [train_ds, dev_ds, test_ds]:
            if ds: ds.samples = ds.samples[:200]

    # --- W&B run (tracking only) ---
    run = None
    if not args.mock:
        hparams = {
            "model": model_cfg["cafebert_name"], "max_length": model_cfg["max_length"],
            "dropout": model_cfg["dropout"], "epochs": tr_cfg["epochs"],
            "early_stopping_patience": tr_cfg.get("early_stopping_patience"), "lr": tr_cfg["lr"],
            "batch_size": tr_cfg["batch_size"], "grad_accum_steps": tr_cfg.get("grad_accum_steps", 1),
            "effective_batch_size": tr_cfg["batch_size"] * tr_cfg.get("grad_accum_steps", 1),
            "fp16": tr_cfg.get("fp16", False), "gradient_checkpointing": model_cfg.get("gradient_checkpointing", False),
            "warmup_ratio": tr_cfg["warmup_ratio"],
            "weight_decay": tr_cfg["weight_decay"], "seed": cfg["project"]["seed"],
            "num_train": len(train_ds), "num_dev": len(dev_ds), "num_test": len(test_ds) if test_ds else 0,
        }
        run = wandb_helper.init_run(cfg, "flat", hparams)

    model = FlatCafeBERT(model_cfg["cafebert_name"], model_cfg["fallback_models"], model_cfg["dropout"], num_labels=3,
                        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False))
    model.to(device)

    grad_accum = max(1, tr_cfg.get("grad_accum_steps", 1))
    fp16_enabled = device.type == "cuda" and tr_cfg.get("fp16", False)
    scaler = torch.amp.GradScaler(device.type, enabled=fp16_enabled)
    print(f"[train_flat] grad_accum_steps={grad_accum} fp16={fp16_enabled} effective_batch={tr_cfg['batch_size']*grad_accum}")

    def collate(batch):
        keys = ["input_ids", "attention_mask"]
        if "token_type_ids" in batch[0]: keys.append("token_type_ids")
        out = {k: torch.stack([b[k] for b in batch]) for k in keys}
        out["labels"] = torch.tensor([b["gold_id"] for b in batch], dtype=torch.long)
        out["sample_id"] = [b["sample_id"] for b in batch]
        out["gold_label"] = [b["gold_label"] for b in batch]
        return out

    num_workers = tr_cfg.get("num_workers", 0)
    dl_kwargs = dict(num_workers=num_workers, pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=tr_cfg["batch_size"], shuffle=True, collate_fn=collate, **dl_kwargs)
    dev_loader   = torch.utils.data.DataLoader(dev_ds, batch_size=tr_cfg["batch_size"] * 2, collate_fn=collate, **dl_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])
    steps_per_epoch = -(-len(train_loader) // grad_accum)  # ceil: 1 optimizer step per grad_accum batches
    total_steps = steps_per_epoch * tr_cfg["epochs"]
    warmup = int(total_steps * tr_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    best_macro = -1; best_path = pathlib.Path(out_cfg["flat_ckpt"]) / "best"
    best_path.mkdir(parents=True, exist_ok=True)
    patience = tr_cfg.get("early_stopping_patience")
    epochs_no_improve = 0

    if args.mock:
        print("[train_flat] MOCK mode — random logits, skip training (no W&B/HF push)")
        return mock_predict(dev_ds, tok, model_cfg, out_cfg, args)

    global_step = 0
    for epoch in range(1, tr_cfg["epochs"] + 1):
        model.train(); total_loss = 0
        optimizer.zero_grad()
        n_batches = len(train_loader)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Flat Epoch {epoch}/{tr_cfg['epochs']}"), start=1):
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k not in ("sample_id", "gold_label")}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16_enabled):
                out = model(input_ids=batch_dev["input_ids"], attention_mask=batch_dev["attention_mask"],
                            token_type_ids=batch_dev.get("token_type_ids"), labels=batch_dev["labels"])
                loss = out["loss"] / grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum

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
                if global_step % tr_cfg["logging_steps"] == 0:
                    wandb_helper.log_step(run, {
                        "train/loss": total_loss / batch_idx,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch_progress": epoch - 1 + batch_idx / n_batches,
                    }, step=global_step)
        train_loss_epoch = total_loss / len(train_loader)
        print(f"[epoch {epoch}] loss={train_loss_epoch:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        metrics, dev_loss, _ = evaluate(model, dev_loader, device, compute_loss=True)
        print(f"[eval dev] {metrics} dev_loss={dev_loss:.4f}")
        wandb_helper.log_epoch_metrics(run, "dev", {**metrics, "loss": dev_loss}, epoch)
        wandb_helper.log_epoch_metrics(run, "train", {"loss": train_loss_epoch}, epoch)

        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path / "pytorch_model.bin")
            tok.save_pretrained(best_path)
            print(f"  ★ new best {best_macro:.4f} -> {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  no improvement ({epochs_no_improve}/{patience or '∞'} epoch(s))")
            if patience is not None and epochs_no_improve >= patience:
                print(f"[train_flat] early stopping — dev macro_f1 không cải thiện sau {patience} epoch liên tiếp (best={best_macro:.4f})")
                wandb_helper.log_step(run, {"train/early_stopped_at_epoch": epoch})
                break

    print("[train_flat] loading best for prediction...")
    try:
        model.load_state_dict(torch.load(best_path / "pytorch_model.bin", map_location=device))
    except Exception as e:
        print(f"[train_flat] could not reload best checkpoint: {e}")

    pred_paths = {}
    final_metrics = {}
    for split, ds, loader in [("dev", dev_ds, dev_loader)] + ([("test", test_ds, torch.utils.data.DataLoader(test_ds, batch_size=tr_cfg["batch_size"] * 2, collate_fn=collate, **dl_kwargs))] if test_ds else []):
        metrics, _, df = evaluate(model, loader, device, return_df=True)
        # save prediction csv per spec: sample_id,gold_label,logit_E,logit_C,logit_N,pred_flat
        out_csv = pathlib.Path(out_cfg["flat_pred"]) / f"{split}_predictions.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        pred_paths[split] = out_csv
        final_metrics[split] = metrics

        y_true = [LABEL2ID_FLAT[x] for x in df["gold_label"]]
        y_pred = df["pred_flat"].tolist()
        _, cm_df, cm, report = diagnostic_report(y_true, y_pred)
        met_path = pathlib.Path(out_cfg["metrics"]) / f"flat_{split}_metrics.json"
        met_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(met_path, "w"), indent=2)
        diag_dir = pathlib.Path(out_cfg["diagnostics"])
        diag_dir.mkdir(parents=True, exist_ok=True)
        cm_df.to_csv(diag_dir / f"flat_{split}_cm.csv")
        open(diag_dir / f"flat_{split}_report.txt", "w").write(report)
        print(f"[predict {split}] -> {out_csv} | metrics={metrics}")

        wandb_helper.log_epoch_metrics(run, f"final_{split}", metrics, tr_cfg["epochs"])
        wandb_helper.log_confusion_matrix(run, y_true, y_pred, ["E", "C", "N"], f"confusion_matrix/{split}")

    # --- MANDATORY storage: push checkpoint + tokenizer + predictions to private HF Hub repo ---
    run_metadata = {
        "run_type": "flat", "model": model.model_name_used if hasattr(model, "model_name_used") else model_cfg["cafebert_name"],
        "seed": cfg["project"]["seed"], "hparams": tr_cfg, "metrics": final_metrics,
        "wandb_run_url": run.url if run else None,
    }
    repo_id, revision = hf_hub_helper.push_run_artifacts(cfg, "flat", best_path, pred_paths, run_metadata)
    if repo_id:
        wandb_helper.log_hf_link(run, repo_id, revision)

    wandb_helper.finish(run)
    return final_metrics


def evaluate(model, loader, device, return_df=False, compute_loss=False):
    model.eval()
    all_logits = []; all_gold = []; all_ids = []; total_loss = 0.0; n_batches = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["sample_id"]; gold = batch["gold_label"]
            tens = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            labels = batch["labels"].to(device) if compute_loss and "labels" in batch else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(**tens, labels=labels)
            if compute_loss and out["loss"] is not None:
                total_loss += out["loss"].item(); n_batches += 1
            logits = out["logits"].cpu().numpy()
            all_logits.append(logits); all_gold.extend(gold); all_ids.extend(ids)
    all_logits = np.concatenate(all_logits, axis=0)
    pred = all_logits.argmax(axis=1)
    y_true = [LABEL2ID_FLAT[g] for g in all_gold]
    metrics = compute_metrics(y_true, pred)
    avg_loss = total_loss / n_batches if n_batches else None
    df = None
    if return_df:
        df = pd.DataFrame({"sample_id": all_ids, "gold_label": all_gold,
                           "logit_E": all_logits[:, 0], "logit_C": all_logits[:, 1], "logit_N": all_logits[:, 2],
                           "pred_flat": pred, "pred_label": [["E", "C", "N"][p] for p in pred]})
    return metrics, avg_loss, df


def mock_predict(dev_ds, tok, model_cfg, out_cfg, args):
    """Generate random but plausible predictions for pipeline verification without training.
    Local-only debug aid — never pushed to W&B/HF (those get real-training runs only)."""
    np.random.seed(42)
    for split in ["dev", "test"]:
        src = f"data/raw/{split}.jsonl"
        import json as _json, pathlib as _pathlib
        if not _pathlib.Path(src).exists(): continue
        rows = []
        with open(src) as f:
            for line in f:
                r = _json.loads(line); rows.append(r)
        data = []
        for r in rows:
            gold = r["label"]; gold_id = LABEL2ID_FLAT[gold]
            logits = np.random.randn(3)
            if np.random.rand() < 0.62: logits[gold_id] += 1.2
            pred = int(logits.argmax())
            data.append({"sample_id": r["id"], "gold_label": gold, "logit_E": float(logits[0]), "logit_C": float(logits[1]), "logit_N": float(logits[2]), "pred_flat": pred, "pred_label": ["E", "C", "N"][pred]})
        df = pd.DataFrame(data)
        out = pathlib.Path(out_cfg["flat_pred"]) / f"{split}_predictions.csv"; out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[mock flat] {out} ({len(df)} rows)")
        from src.evaluation.metrics import diagnostic_report
        y_true = [LABEL2ID_FLAT[g] for g in df["gold_label"]]; y_pred = df["pred_flat"].tolist()
        mets, cm_df, cm, rep = diagnostic_report(y_true, y_pred)
        met_path = pathlib.Path(out_cfg["metrics"]) / f"flat_{split}_metrics.json"; met_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(mets, open(met_path, "w"), indent=2)
        diag = pathlib.Path(out_cfg["diagnostics"]); diag.mkdir(parents=True, exist_ok=True)
        cm_df.to_csv(diag / f"flat_{split}_cm.csv")
        open(diag / f"flat_{split}_report.txt", "w").write(rep)
    print("[mock] done. Run scripts/evaluate_offline.py next.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mock", action="store_true", help="skip training, generate random predictions for pipeline test (local only, no W&B/HF)")
    args = ap.parse_args()
    train(args)
