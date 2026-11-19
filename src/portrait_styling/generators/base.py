"""Base abstractions for image-editing model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationRequest:
    """Normalized request passed to any generator backend."""

    portrait_path: Path
    prompt: str
    output_path: Path
    seed: int = 42
    negative_prompt: str = ""
    reference_paths: list[Path] = field(default_factory=list)
    width: int = 768
    height: int = 1024
    steps: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationOutput:
    """Normalized output returned by any generator backend."""

    image_path: Path
    latency_sec: float
    seed: int
    model_alias: str
    model_id: str
    prompt_used: str


class BaseImageEditor(ABC):
    """Abstract interface for both local/Colab diffusers and remote colabllm backends."""

    def __init__(self, alias: str, config: dict[str, Any]) -> None:
        self.alias = alias
        self.config = config
        self.model_id: str = str(config.get("model_id", alias))
        if not config.get("sfw_only", True):
            raise ValueError(f"Model '{alias}' is not marked sfw_only=true; refusing to load.")

    @property
    def preferred_language(self) -> str:
        """Return 'en' for English-first models (FLUX), otherwise 'zh'."""
        if "flux" in self.alias.lower() or "flux" in self.model_id.lower():
            return "en"
        return "zh"

    @abstractmethod
    def edit(self, request: GenerationRequest) -> GenerationOutput:
        """Run a single portrait editing pass and save the output PNG."""

    def unload(self) -> None:
        """Optional hook to release GPU VRAM before loading another model in A/B runs."""
        return None
