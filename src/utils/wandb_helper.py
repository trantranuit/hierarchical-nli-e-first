"""W&B helper — TRACKING ONLY.

Logs hyperparameters, seed, lr, batch_size, train/dev loss, Accuracy, Macro-F1,
F1 per class, confusion matrix, and run metadata (incl. the HF Hub repo_id/revision
where the actual checkpoint/tokenizer/predictions are stored).

W&B never receives checkpoints, tokenizer files, or per-sample predictions —
those are mandatory-stored on Hugging Face Hub instead (see hf_hub_helper.py).
"""
from __future__ import annotations
import os
from src.utils.env import load_env  # noqa: F401 (import side-effect: loads .env)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def wandb_enabled(cfg: dict) -> bool:
    return HAS_WANDB and cfg.get("wandb", {}).get("enabled", True) and bool(os.getenv("WANDB_API_KEY"))


def init_run(cfg: dict, run_type: str, hparams: dict, tags=None):
    """run_type: 'flat' or 'hier'. Returns the wandb run object, or None if disabled."""
    if not wandb_enabled(cfg):
        print(f"[wandb] disabled/not configured — skipping tracking for {run_type}")
        return None
    wb_cfg = cfg.get("wandb", {})
    exp_key = "experiment_flat" if run_type == "flat" else "experiment_hier"
    try:
        run = wandb.init(
            project=wb_cfg.get("project", "hierarchical-nli-e-first"),
            entity=wb_cfg.get("entity") or None,
            name=f"{wb_cfg.get(exp_key, run_type)}-seed{cfg['project']['seed']}",
            group=wb_cfg.get(exp_key, run_type),
            job_type=run_type,
            tags=tags or [run_type, cfg["project"].get("version", "v1")],
            config=hparams,
        )
        print(f"[wandb] run started: {run.url}")
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}) — continuing without tracking")
        return None


def log_step(run, metrics: dict, step: int | None = None):
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as e:
        print(f"[wandb] log_step failed: {e}")


def log_epoch_metrics(run, split: str, metrics: dict, epoch: int):
    """metrics: dict with accuracy, macro_f1, f1_E, f1_C, f1_N, loss, ..."""
    if run is None:
        return
    payload = {f"{split}/{k}": v for k, v in metrics.items()}
    payload["epoch"] = epoch
    log_step(run, payload)


def log_confusion_matrix(run, y_true, y_pred, class_names, title: str):
    if run is None:
        return
    try:
        run.log({title: wandb.plot.confusion_matrix(
            preds=list(y_pred), y_true=list(y_true), class_names=class_names)})
    except Exception as e:
        print(f"[wandb] confusion matrix log failed: {e}")


def log_hf_link(run, repo_id: str, revision: str, artifact_type: str = "model"):
    """Log the HF Hub repo_id/revision for this run so the exact checkpoint can be traced."""
    if run is None:
        return
    try:
        run.summary["hf_repo_id"] = repo_id
        run.summary["hf_revision"] = revision
        run.summary["hf_url"] = f"https://huggingface.co/{repo_id}/tree/{revision}"
        run.log({"hf_repo_id": repo_id, "hf_revision": revision})
    except Exception as e:
        print(f"[wandb] log_hf_link failed: {e}")


def finish(run):
    if run is None:
        return
    try:
        run.finish()
        print("[wandb] run closed")
    except Exception as e:
        print(f"[wandb] finish failed: {e}")
