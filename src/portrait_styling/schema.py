"""Typed data schemas for the 4-stage portrait styling pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HairPlan:
    length: str = ""
    color: str = ""
    bangs: str = ""
    texture: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HairPlan":
        if not data:
            return cls()
        return cls(
            length=str(data.get("length", "") or ""),
            color=str(data.get("color", "") or ""),
            bangs=str(data.get("bangs", "") or ""),
            texture=str(data.get("texture", "") or ""),
        )


@dataclass
class MakeupPlan:
    base: str = ""
    eyebrows: str = ""
    eyes: str = ""
    blush: str = ""
    contour: str = ""
    lips: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MakeupPlan":
        if not data:
            return cls()
        return cls(
            base=str(data.get("base", "") or ""),
            eyebrows=str(data.get("eyebrows", "") or ""),
            eyes=str(data.get("eyes", "") or ""),
            blush=str(data.get("blush", "") or ""),
            contour=str(data.get("contour", "") or ""),
            lips=str(data.get("lips", "") or ""),
        )


@dataclass
class DiagnosisInput:
    """Upstream structured portrait analysis & expert styling recommendation."""

    sample_id: str = "sample_001"
    gender: str = "female"
    face_shape: str = ""
    facial_thirds: dict[str, str] = field(default_factory=dict)
    facial_fifths: str = ""
    skin_tone: str = ""
    style_goal: str = ""
    hair_plan: HairPlan = field(default_factory=HairPlan)
    makeup_plan: MakeupPlan = field(default_factory=MakeupPlan)
    lighting: str = "自然室内柔光"
    custom_prompt_override: str = ""
    reference_images: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiagnosisInput":
        return cls(
            sample_id=str(data.get("sample_id", "sample_001")),
            gender=str(data.get("gender", "female")),
            face_shape=str(data.get("face_shape", "") or ""),
            facial_thirds=dict(data.get("facial_thirds") or {}),
            facial_fifths=str(data.get("facial_fifths", "") or ""),
            skin_tone=str(data.get("skin_tone", "") or ""),
            style_goal=str(data.get("style_goal", "") or ""),
            hair_plan=HairPlan.from_dict(data.get("hair_plan")),
            makeup_plan=MakeupPlan.from_dict(data.get("makeup_plan")),
            lighting=str(data.get("lighting", "自然室内柔光") or "自然室内柔光"),
            custom_prompt_override=str(data.get("custom_prompt_override", "") or ""),
            reference_images=dict(data.get("reference_images") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessedPortrait:
    """Output of Stage 1: aligned portrait image + optional ArcFace identity anchor."""

    source_path: str
    cropped_path: str
    width: int
    height: int
    face_detected: bool
    face_bbox: list[float] | None = None
    embedding: list[float] | None = None
    detector_backend: str = "none"


@dataclass
class TranslatedPrompt:
    """Output of Stage 2: executable visual prompt + checklist for VLM scoring."""

    prompt_zh: str
    prompt_en: str
    negative_prompt: str
    checklist: list[str] = field(default_factory=list)
    translator_mode: str = "rule"

    def select_prompt(self, language: str = "zh") -> str:
        if language.lower().startswith("en") and self.prompt_en:
            return self.prompt_en
        return self.prompt_zh


@dataclass
class CandidateEval:
    """Output of Stage 4: ID preservation gate + optional VLM rule adherence score."""

    id_similarity: float | None = None
    id_passed: bool | None = None
    vlm_score: float | None = None
    vlm_details: dict[str, Any] = field(default_factory=dict)
    composite_score: float = 0.0
    notes: str = ""


@dataclass
class GeneratedCandidate:
    """Single generated candidate for a (sample, model, seed) tuple."""

    model_alias: str
    model_id: str
    seed: int
    image_path: str
    latency_sec: float
    prompt_used: str
    eval_result: CandidateEval = field(default_factory=CandidateEval)
    error: str | None = None


@dataclass
class SampleModelRun:
    """Aggregated Best-of-N results for one sample on one model."""

    sample_id: str
    model_alias: str
    model_id: str
    best_candidate: GeneratedCandidate | None
    candidates: list[GeneratedCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRecord:
    """Complete benchmark record for one portrait across all evaluated models."""

    sample_id: str
    source_image: str
    preprocessed_image: str
    diagnosis: dict[str, Any]
    translated_prompt: dict[str, Any]
    model_runs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_optional_path(path_value: str | None, base_dir: Path | None = None) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p
