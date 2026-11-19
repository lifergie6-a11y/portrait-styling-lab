#!/usr/bin/env python3
"""Scan a folder of consented test portraits and create a golden benchmark manifest.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_TEMPLATES = [
    {
        "face_shape": "方圆脸",
        "facial_thirds": {"upper": "标准", "middle": "偏长", "lower": "标准"},
        "skin_tone": "秋季暖皮",
        "style_goal": "柔和颧骨、自然缩短中庭",
        "hair_plan": {
            "length": "锁骨层次中长发",
            "color": "深暖茶棕色",
            "bangs": "轻薄八字刘海自然遮挡两侧颧骨",
            "texture": "发根自然蓬松、发尾微内扣",
        },
        "makeup_plan": {
            "base": "轻透自然雾面底妆，保留真实肤质与毛孔",
            "eyebrows": "柔和微弧自然野生眉",
            "eyes": "自然卧蚕与柔和眼尾微下至眼线，低饱和暖棕大地色眼影",
            "blush": "眼下苹果肌横向轻扫低饱和蜜桃色腮红",
            "lips": "低饱和肉桂奶茶色哑光唇釉",
        },
        "lighting": "自然室内柔光",
    },
    {
        "face_shape": "圆脸",
        "facial_thirds": {"upper": "偏短", "middle": "标准", "lower": "偏短"},
        "skin_tone": "夏季冷皮",
        "style_goal": "拉长纵向比例、清冷知性质感",
        "hair_plan": {
            "length": "中长直发带微层次",
            "color": "冷调深灰棕色",
            "bangs": "轻盈侧分微碎发修饰额角",
            "texture": "高颅顶自然蓬松，线条利落柔顺",
        },
        "makeup_plan": {
            "base": "清透微光泽奶油肌底妆，保留自然肤理",
            "eyebrows": "细长微挑灰棕色自然眉",
            "eyes": "低饱和灰粉大地色眼影，自然纤长睫毛与细内眼线",
            "blush": "颧骨外侧斜向轻扫低饱和冷调玫瑰粉腮红",
            "lips": "低饱和玫瑰豆沙色半哑光口红",
        },
        "lighting": "自然柔光",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a golden benchmark manifest.json from an image folder.")
    parser.add_argument("--images-dir", required=True, help="Directory containing portrait images")
    parser.add_argument("--out", required=True, help="Output manifest.json path")
    args = parser.parse_args()

    img_dir = Path(args.images_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    samples = []
    for idx, img_path in enumerate(images):
        try:
            rel = str(img_path.relative_to(out_path.parent))
        except ValueError:
            rel = str(img_path)
        tpl = DEFAULT_TEMPLATES[idx % len(DEFAULT_TEMPLATES)]
        samples.append(
            {
                "sample_id": f"s{idx + 1:02d}_{img_path.stem}",
                "image": rel,
                "diagnosis": {
                    "sample_id": f"s{idx + 1:02d}_{img_path.stem}",
                    "gender": "female",
                    **tpl,
                },
            }
        )

    out_path.write_text(
        json.dumps({"version": "1.0", "samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} sample entries to {out_path}")


if __name__ == "__main__":
    main()
