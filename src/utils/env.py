"""Load .env (WANDB_API_KEY, HF_TOKEN) once, at import time, without ever printing the values."""
from __future__ import annotations
import os
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env", override=False)
    except ImportError:
        # fallback: parse manually if python-dotenv isn't installed
        env_path = _ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()
