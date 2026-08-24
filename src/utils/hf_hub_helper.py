"""Hugging Face Hub helper — MANDATORY storage for reproducibility.

Everything needed to reload a model and rerun Hard/Soft inference, error analysis,
or reproduce results without retraining lives in a PRIVATE HF Hub repo per run:
  - best checkpoint (pytorch_model.bin) + tokenizer/config
  - dev/test predictions per sample (logits), matching the exact spec:
      Flat:  sample_id, gold_label, logit_E, logit_C, logit_N
      Hier:  sample_id, gold_label, + coarse_logit_*/fine_logit_* (cột đúng tên tùy head_design —
             e_first: coarse_logit_E/nonE, fine_logit_C/N | n_first: coarse_logit_N/nonN, fine_logit_E/C)

W&B only stores metrics/metadata + a pointer (repo_id/revision) back here.
"""
from __future__ import annotations
import os
import pathlib
from src.utils.env import load_env  # noqa: F401

try:
    from huggingface_hub import HfApi
    HAS_HF = True
except ImportError:
    HAS_HF = False


def hf_enabled(cfg: dict) -> bool:
    return HAS_HF and cfg.get("hf_hub", {}).get("enabled", True) and bool(os.getenv("HF_TOKEN"))


def _api():
    return HfApi(token=os.environ["HF_TOKEN"])


def ensure_repo(repo_id: str, private: bool = True):
    api = _api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return repo_id


def _next_version_tag(api, repo_id: str) -> str:
    """v1, v2, v3, ... — same repo every run, no new repo names. Looks at existing
    git tags on the repo and picks the next free vN."""
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
        nums = [int(r.name[1:]) for r in refs.tags if r.name.startswith("v") and r.name[1:].isdigit()]
        return f"v{(max(nums) + 1) if nums else 1}"
    except Exception as e:
        print(f"[hf_hub] could not list existing tags ({e}) — defaulting to v1")
        return "v1"


def push_run_artifacts(cfg: dict, run_type: str, checkpoint_dir: pathlib.Path,
                        pred_paths: dict, run_metadata: dict) -> tuple[str, str] | tuple[None, None]:
    """Upload checkpoint dir + prediction CSVs (dev/test) to the run's private HF repo.

    pred_paths: {"dev": Path, "test": Path} pointing at the per-sample logits CSV.
    Every run pushes to the SAME repo (configs/config.yaml -> hf_hub.{run_type}_repo_id) —
    no new repo name per run. Instead, each push is tagged v1, v2, v3, ... on that repo,
    so past versions stay retrievable (e.g. `from_pretrained(repo_id, revision="v2")`)
    even after `main` moves on to a newer push.
    Returns (repo_id, version_tag) — version_tag is "v1"/"v2"/... (falls back to the raw
    commit SHA if tag creation fails). Returns (None, None) if HF Hub push is disabled.
    """
    if not hf_enabled(cfg):
        print(f"[hf_hub] disabled/not configured — skipping upload for {run_type}")
        return None, None

    hf_cfg = cfg.get("hf_hub", {})
    repo_id = hf_cfg.get(f"{run_type}_repo_id")
    if not repo_id:
        print(f"[hf_hub] no repo_id configured for {run_type} — skipping upload")
        return None, None

    private = hf_cfg.get("private", True)
    ensure_repo(repo_id, private=private)
    api = _api()

    commit_msg = f"{run_type} run — seed={run_metadata.get('seed')} lambda={run_metadata.get('lambda_fine', 'n/a')}"

    # 1) checkpoint + tokenizer/config
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        api.upload_folder(
            repo_id=repo_id, repo_type="model",
            folder_path=str(checkpoint_dir),
            path_in_repo="checkpoint",
            commit_message=f"{commit_msg} — checkpoint",
        )
    else:
        print(f"[hf_hub] checkpoint dir empty, skipping checkpoint upload ({checkpoint_dir})")

    # 2) predictions (per-sample logits, dev/test)
    for split, path in pred_paths.items():
        path = pathlib.Path(path)
        if path.exists():
            api.upload_file(
                repo_id=repo_id, repo_type="model",
                path_or_fileobj=str(path),
                path_in_repo=f"predictions/{path.name}",
                commit_message=f"{commit_msg} — {split} predictions",
            )
        else:
            print(f"[hf_hub] predictions file missing, skipping: {path}")

    # 3) run metadata as a small JSON card (seed, hparams, metrics pointer)
    import json, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)
        meta_path = f.name
    commit_info = api.upload_file(
        repo_id=repo_id, repo_type="model",
        path_or_fileobj=meta_path,
        path_in_repo="run_metadata.json",
        commit_message=f"{commit_msg} — run metadata",
    )
    os.unlink(meta_path)

    commit_sha = getattr(commit_info, "oid", None) or "main"

    # tag this commit as the next version (v1, v2, v3, ...) — same repo, no new repo names
    version_tag = _next_version_tag(api, repo_id)
    try:
        api.create_tag(repo_id=repo_id, tag=version_tag, revision=commit_sha, repo_type="model",
                       tag_message=f"{run_type} run — seed={run_metadata.get('seed')}")
        print(f"[hf_hub] tagged commit {commit_sha} as {version_tag}")
    except Exception as e:
        print(f"[hf_hub] tag creation failed ({e}) — falling back to raw commit SHA as revision")
        version_tag = commit_sha

    print(f"[hf_hub] pushed {run_type} run -> https://huggingface.co/{repo_id} @ {version_tag} (commit {commit_sha})")
    return repo_id, version_tag
