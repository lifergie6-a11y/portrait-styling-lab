"""Stage 4: Automated Identity Gate (ArcFace cosine similarity) & VLM Rule-Adherence Scorer."""

from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from .preprocess import FaceAnchorEngine
from .schema import CandidateEval, GeneratedCandidate

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: list[float] | np.ndarray, vec_b: list[float] | np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _encode_image_data_url(image_path: str | Path, max_side: int = 768) -> str:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class AutoScorer:
    """Evaluates generated portrait edits using ID similarity gate + optional VLM judge."""

    def __init__(
        self,
        face_engine: FaceAnchorEngine | None = None,
        id_threshold: float = 0.65,
        vlm_base_url: str | None = None,
        vlm_api_key: str | None = None,
        vlm_model: str | None = None,
        vlm_timeout: int = 90,
    ) -> None:
        self.face_engine = face_engine
        self.id_threshold = id_threshold
        self.vlm_base_url = (vlm_base_url or os.getenv("STYLING_VLM_BASE_URL", "")).rstrip("/")
        self.vlm_api_key = vlm_api_key or os.getenv("STYLING_VLM_API_KEY", "")
        self.vlm_model = vlm_model or os.getenv("STYLING_VLM_MODEL", "Qwen3.8-Flash-Next")
        self.vlm_timeout = vlm_timeout

    def score_id(
        self,
        source_embedding: list[float] | None,
        candidate_path: str | Path,
    ) -> tuple[float | None, bool | None, str]:
        if source_embedding is None or self.face_engine is None:
            return None, None, "ID scorer unavailable (install insightface for ArcFace scoring)"

        img = Image.open(candidate_path).convert("RGB")
        _, cand_emb = self.face_engine.analyze_image(img)
        if cand_emb is None:
            return 0.0, False, "No face detected in edited output"

        sim = round(cosine_similarity(source_embedding, cand_emb), 4)
        passed = sim >= self.id_threshold
        note = "ID pass" if passed else f"ID below threshold ({sim:.3f} < {self.id_threshold:.2f})"
        return sim, passed, note

    def score_vlm(
        self,
        original_path: str | Path,
        candidate_path: str | Path,
        checklist: list[str],
    ) -> tuple[float | None, dict[str, Any]]:
        """Optional OpenAI-compatible Vision-LLM judge for rule adherence and natural skin realism."""
        if not self.vlm_base_url:
            return None, {}

        headers = {"Content-Type": "application/json"}
        if self.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.vlm_api_key}"

        checklist_text = "\n".join(f"- {item}" for item in checklist)
        prompt = (
            "You are a strict portrait hair & makeup QA reviewer. Compare Image 1 (original) and "
            "Image 2 (edited). Evaluate whether Image 2 preserves the person's identity, avoids "
            "plastic/over-smoothed filter skin, blends the hairline naturally, and satisfies:\n"
            f"{checklist_text}\n\n"
            "Return strict JSON with keys:\n"
            '  "overall_score": float from 0.0 to 10.0,\n'
            '  "naturalness": float from 0.0 to 10.0,\n'
            '  "rule_adherence": float from 0.0 to 10.0,\n'
            '  "brief_comment": short one-sentence critique in Chinese.'
        )

        payload = {
            "model": self.vlm_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _encode_image_data_url(original_path)}},
                        {"type": "image_url", "image_url": {"url": _encode_image_data_url(candidate_path)}},
                    ],
                }
            ],
        }

        try:
            resp = requests.post(
                f"{self.vlm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.vlm_timeout,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            overall = float(parsed.get("overall_score", 0.0))
            return round(overall, 2), parsed
        except Exception as exc:
            logger.warning("VLM scoring failed (%s); continuing without VLM score.", exc)
            return None, {"error": str(exc)}

    def evaluate_candidate(
        self,
        original_path: str | Path,
        source_embedding: list[float] | None,
        candidate_path: str | Path,
        checklist: list[str],
    ) -> CandidateEval:
        id_sim, id_passed, note = self.score_id(source_embedding, candidate_path)
        vlm_score, vlm_details = self.score_vlm(original_path, candidate_path, checklist)

        # Composite score (0..100 scale): hard gate penalty if ID check fails
        id_component = (id_sim * 100.0) if id_sim is not None else 50.0
        if vlm_score is not None:
            vlm_component = vlm_score * 10.0
            composite = 0.55 * id_component + 0.45 * vlm_component
        else:
            composite = id_component

        if id_passed is False:
            composite *= 0.4  # Strongly demote identity-drifted outputs

        return CandidateEval(
            id_similarity=id_sim,
            id_passed=id_passed,
            vlm_score=vlm_score,
            vlm_details=vlm_details,
            composite_score=round(composite, 2),
            notes=note,
        )


def select_best_candidate(candidates: list[GeneratedCandidate]) -> GeneratedCandidate | None:
    """Pick the highest-scoring valid candidate, preferring those that pass the ID gate."""
    valid = [c for c in candidates if not c.error and c.image_path]
    if not valid:
        return None
    valid.sort(
        key=lambda c: (
            1 if c.eval_result.id_passed is not False else 0,
            c.eval_result.composite_score,
            c.eval_result.id_similarity if c.eval_result.id_similarity is not None else -1.0,
        ),
        reverse=True,
    )
    return valid[0]
