#!/usr/bin/env python3
"""Pre-download SFW model weights into a persistent Google Drive Hugging Face cache.

Best practice for Google Colab Ultra / Pro:
Run this script on a FREE CPU runtime (0 CCU/h) first so you don't burn GPU compute units
waiting for 20-40 GB model downloads.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from portrait_styling.generators import load_model_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download SFW model weights on CPU (0 CCU/h).")
    parser.add_argument(
        "--models",
        default="firered-1.1,qwen-2.1",
        help="Comma-separated model aliases from configs/models.yaml",
    )
    parser.add_argument(
        "--cache-dir",
        default="/content/drive/MyDrive/portrait-styling-lab/hf_cache",
        help="Persistent Hugging Face cache directory",
    )
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Optional HF token")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download  # type: ignore

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    registry = load_model_registry()

    for alias in [m.strip() for m in args.models.split(",") if m.strip()]:
        if alias not in registry:
            raise SystemExit(f"Unknown model alias '{alias}'.")
        cfg = registry[alias]
        if cfg.get("backend") != "diffusers":
            print(f"[skip] '{alias}' uses backend={cfg.get('backend')} (no local HF snapshot needed).")
            continue
        model_id = cfg["model_id"]
        print(f"Downloading {alias} -> {model_id} into {cache_dir} ...")
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            token=args.hf_token,
        )
        print(f"Cached {model_id} at {local_dir}")


if __name__ == "__main__":
    main()
