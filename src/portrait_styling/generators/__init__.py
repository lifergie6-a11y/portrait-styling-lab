"""Generator factory and model registry loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import BaseImageEditor, GenerationOutput, GenerationRequest
from .colabllm_backend import ColabLLMImageEditor
from .diffusers_backend import DiffusersImageEditor

DEFAULT_MODELS_YAML = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


def load_model_registry(config_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    path = Path(config_path) if config_path else DEFAULT_MODELS_YAML
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = raw.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Invalid models.yaml format in {path}: expected top-level 'models' mapping.")
    return models


def create_generator(
    model_alias: str,
    registry: dict[str, dict[str, Any]] | None = None,
    quantization: str = "auto",
    cpu_offload: str = "auto",
    cache_dir: str | None = None,
    colabllm_instance: str | None = None,
    timeout: int = 300,
) -> BaseImageEditor:
    reg = registry if registry is not None else load_model_registry()
    if model_alias not in reg:
        available = ", ".join(sorted(reg.keys()))
        raise KeyError(f"Unknown model alias '{model_alias}'. Available: {available}")

    cfg = reg[model_alias]
    backend = str(cfg.get("backend", "")).strip()
    if backend == "diffusers":
        return DiffusersImageEditor(
            alias=model_alias,
            config=cfg,
            quantization=quantization,
            cpu_offload=cpu_offload,
            cache_dir=cache_dir,
        )
    if backend == "colabllm":
        return ColabLLMImageEditor(
            alias=model_alias,
            config=cfg,
            instance=colabllm_instance,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported backend '{backend}' for model '{model_alias}'.")


__all__ = [
    "BaseImageEditor",
    "ColabLLMImageEditor",
    "DiffusersImageEditor",
    "GenerationOutput",
    "GenerationRequest",
    "create_generator",
    "load_model_registry",
]
