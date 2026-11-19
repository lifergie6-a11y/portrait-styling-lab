"""Stage 2: Translate structured expert diagnosis JSON into executable SFW visual editing prompts.

Avoids feeding abstract diagnostic jargon (e.g., "长中庭、秋季暖皮、显嫩") directly to image models.
Instead translates them into concrete, observable hair, makeup, and lighting instructions while
locking facial bone structure, identity, and authentic skin texture.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

from .schema import DiagnosisInput, TranslatedPrompt

logger = logging.getLogger(__name__)

# Strictly SFW guardrail keywords that must never be passed to editing models
_BLOCKED_TERMS = (
    "nsfw",
    "nude",
    "naked",
    "lingerie",
    "topless",
    "bikini",
    "undress",
    "cleavage",
    "erotic",
    "sexy lingerie",
    "裸露",
    "色情",
    "情趣",
    "脱衣",
    "内衣",
)

DEFAULT_NEGATIVE_PROMPT_ZH = (
    "换脸，五官位移，过度磨皮，假面感，网红锥子脸，塑料皮肤，畸变，模糊，低画质，多余发饰"
)

DEFAULT_IDENTITY_LOCK_ZH = (
    "严格保持图中人物原本的五官位置、眼睛形状、鼻型、唇形、脸型轮廓、神态与姿势不变。"
)

DEFAULT_TEXTURE_LOCK_ZH = (
    "保留真实自然的皮肤纹理与轻微毛孔质感，拒绝过度磨皮、假面厚粉感或网红脸变形；"
    "发际线与头骨贴合自然，整体光线与原环境融合一致，保持日常得体的半身人像。"
)

DEFAULT_IDENTITY_LOCK_EN = (
    "Strictly preserve the person's original facial identity, eye shape, nose, mouth proportions, "
    "bone structure, expression, and pose."
)

DEFAULT_TEXTURE_LOCK_EN = (
    "Maintain authentic natural skin texture and subtle pores without heavy airbrushing, plastic skin, "
    "or over-sharpened beauty filters. Ensure realistic hairline blending, natural indoor lighting consistency, "
    "and tasteful everyday portrait framing."
)


def assert_sfw_text(text: str) -> None:
    """Reject any non-SFW prompt text to enforce strict SFW-only operation."""
    lowered = text.lower()
    for term in _BLOCKED_TERMS:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered) if term.isascii() else term in lowered:
            raise ValueError(
                f"SFW guardrail triggered: disallowed term '{term}' detected in styling prompt."
            )


def _derive_default_adjustments(diag: DiagnosisInput) -> tuple[list[str], list[str]]:
    """Infer concrete hair/makeup visual cues if individual fields are left sparse."""
    hair_cues: list[str] = []
    makeup_cues: list[str] = []

    face_shape = diag.face_shape
    if "方" in face_shape and not diag.hair_plan.bangs:
        hair_cues.append("配轻薄八字刘海与耳侧碎发自然柔和颧骨及下颌线条")
    elif "圆" in face_shape and not diag.hair_plan.texture:
        hair_cues.append("高颅顶自然蓬松发根，微层次披发拉长纵向比例")
    elif "长" in face_shape and not diag.hair_plan.bangs:
        hair_cues.append("配轻盈空气感法式刘海缩短上庭视觉，两侧发丝微卷增加横向丰盈度")

    middle_third = diag.facial_thirds.get("middle", "")
    if "长" in middle_third and not diag.makeup_plan.blush:
        makeup_cues.append("在眼下苹果肌位置横向轻扫低饱和柔和腮红，并点缀自然卧蚕以缩短中庭视觉")

    skin = diag.skin_tone
    if "暖" in skin and not diag.makeup_plan.lips:
        makeup_cues.append("低饱和暖调奶茶或肉桂豆沙色自然唇妆")
    elif "冷" in skin and not diag.makeup_plan.lips:
        makeup_cues.append("低饱和清冷玫瑰豆沙或浅梅子色水光唇妆")

    return hair_cues, makeup_cues


def translate_rule_based(diag: DiagnosisInput) -> TranslatedPrompt:
    """Deterministic zero-cost translation from DiagnosisInput -> TranslatedPrompt."""
    if diag.custom_prompt_override.strip():
        override = diag.custom_prompt_override.strip()
        assert_sfw_text(override)
        return TranslatedPrompt(
            prompt_zh=override,
            prompt_en=override,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT_ZH,
            checklist=["保持人物身份不变", "整体妆发符合自定义指令", "肤质自然不假面"],
            translator_mode="override",
        )

    fallback_hair, fallback_makeup = _derive_default_adjustments(diag)

    hair_parts = [
        p
        for p in [
            diag.hair_plan.color,
            diag.hair_plan.length,
            diag.hair_plan.bangs,
            diag.hair_plan.texture,
            *fallback_hair,
        ]
        if p
    ]
    makeup_parts = [
        p
        for p in [
            diag.makeup_plan.base,
            diag.makeup_plan.eyebrows,
            diag.makeup_plan.eyes,
            diag.makeup_plan.blush,
            diag.makeup_plan.contour,
            diag.makeup_plan.lips,
            *fallback_makeup,
        ]
        if p
    ]

    ref_notes_zh: list[str] = []
    ref_idx = 2
    if diag.reference_images.get("hair"):
        ref_notes_zh.append(f"发型廓形与层次参考图{ref_idx} <image{ref_idx}>")
        ref_idx += 1
    if diag.reference_images.get("makeup"):
        ref_notes_zh.append(f"妆容色调与质感参考图{ref_idx} <image{ref_idx}>")
        ref_idx += 1

    sections_zh = [DEFAULT_IDENTITY_LOCK_ZH]
    if ref_notes_zh:
        sections_zh.append("以图1 <image1> 为主体人物，" + "，".join(ref_notes_zh) + "。")
    if hair_parts:
        sections_zh.append("发型调整为：" + "，".join(hair_parts) + "。")
    if makeup_parts:
        sections_zh.append("妆容调整为：" + "，".join(makeup_parts) + "。")
    if diag.lighting:
        sections_zh.append(f"光影保持{diag.lighting}。")
    sections_zh.append(DEFAULT_TEXTURE_LOCK_ZH)

    prompt_zh = "".join(sections_zh)
    assert_sfw_text(prompt_zh)

    # Compact English mirror for English-first models (e.g., FLUX Klein)
    en_parts = [DEFAULT_IDENTITY_LOCK_EN]
    if hair_parts:
        en_parts.append(f"Hairstyle edit: {', '.join(hair_parts)}.")
    if makeup_parts:
        en_parts.append(f"Natural makeup edit: {', '.join(makeup_parts)}.")
    if diag.lighting:
        en_parts.append(f"Lighting: {diag.lighting}.")
    en_parts.append(DEFAULT_TEXTURE_LOCK_EN)
    prompt_en = " ".join(en_parts)
    assert_sfw_text(prompt_en)

    checklist: list[str] = ["人物五官与脸型与原图高度一致（无换头感）"]
    if diag.hair_plan.color:
        checklist.append(f"发色准确落实为：{diag.hair_plan.color}")
    if diag.hair_plan.length or diag.hair_plan.bangs:
        checklist.append(
            f"发型与刘海自然贴合头骨：{diag.hair_plan.length} {diag.hair_plan.bangs}".strip()
        )
    if diag.makeup_plan.eyes:
        checklist.append(f"眼妆自然落实：{diag.makeup_plan.eyes}")
    if diag.makeup_plan.blush:
        checklist.append(f"腮红位置与色调准确：{diag.makeup_plan.blush}")
    if diag.makeup_plan.lips:
        checklist.append(f"唇色与质地准确：{diag.makeup_plan.lips}")
    checklist.append("保留真实皮肤纹理与毛孔，无过度磨皮假面感")

    return TranslatedPrompt(
        prompt_zh=prompt_zh,
        prompt_en=prompt_en,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT_ZH,
        checklist=checklist,
        translator_mode="rule",
    )


def translate_with_llm(
    diag: DiagnosisInput,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int = 60,
) -> TranslatedPrompt:
    """Optional LLM translation layer using any OpenAI-compatible chat endpoint.

    Falls back gracefully to `translate_rule_based` if credentials are not configured
    or if the API call fails.
    """
    base_url = (base_url or os.getenv("STYLING_LLM_BASE_URL", "")).rstrip("/")
    api_key = api_key or os.getenv("STYLING_LLM_API_KEY", "")
    model = model or os.getenv("STYLING_LLM_MODEL", "Qwen3.8-Flash-Next")

    if not base_url:
        return translate_rule_based(diag)

    rule_baseline = translate_rule_based(diag)
    system_prompt = (
        "You are a senior SFW portrait hair & makeup visual prompt engineer. "
        "Convert the structured beauty diagnosis JSON into concrete, observable, photorealistic "
        "image-editing instructions in BOTH Chinese (`prompt_zh`) and English (`prompt_en`), plus "
        "a concise verification `checklist` (4-6 items). "
        "NEVER alter facial bone structure or identity. NEVER include NSFW or clothing-removal cues. "
        "Return strict JSON with keys: prompt_zh, prompt_en, checklist."
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(diag.to_dict(), ensure_ascii=False)},
        ],
    }

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        prompt_zh = str(parsed.get("prompt_zh") or rule_baseline.prompt_zh).strip()
        prompt_en = str(parsed.get("prompt_en") or rule_baseline.prompt_en).strip()
        checklist = [str(x) for x in (parsed.get("checklist") or rule_baseline.checklist)]
        assert_sfw_text(prompt_zh)
        assert_sfw_text(prompt_en)
        return TranslatedPrompt(
            prompt_zh=prompt_zh,
            prompt_en=prompt_en,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT_ZH,
            checklist=checklist,
            translator_mode=f"llm:{model}",
        )
    except Exception as exc:
        logger.warning("LLM prompt translation failed (%s); falling back to rule-based prompt.", exc)
        return rule_baseline
