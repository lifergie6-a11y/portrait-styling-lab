import pytest

from portrait_styling.prompt_translator import assert_sfw_text, translate_rule_based
from portrait_styling.schema import DiagnosisInput


def test_rule_based_translation_includes_concrete_cues():
    diag = DiagnosisInput.from_dict(
        {
            "sample_id": "t1",
            "face_shape": "方圆脸",
            "facial_thirds": {"middle": "偏长"},
            "skin_tone": "秋季暖皮",
            "hair_plan": {"color": "深暖茶棕色", "length": "锁骨层次中长发", "bangs": "轻薄八字刘海"},
            "makeup_plan": {"lips": "低饱和肉桂奶茶色哑光唇釉", "blush": "眼下横向轻扫蜜桃色腮红"},
        }
    )
    res = translate_rule_based(diag)
    assert "深暖茶棕色" in res.prompt_zh
    assert "轻薄八字刘海" in res.prompt_zh
    assert "严格保持图中人物原本的五官位置" in res.prompt_zh
    assert "Strictly preserve the person's original facial identity" in res.prompt_en
    assert len(res.checklist) >= 4


def test_sfw_guardrail_blocks_disallowed_terms():
    with pytest.raises(ValueError, match="SFW guardrail"):
        assert_sfw_text("change to bikini outfit")

    with pytest.raises(ValueError, match="SFW guardrail"):
        assert_sfw_text("换成裸露风格")

    # Safe beauty terms should pass
    assert_sfw_text("自然裸妆风格，低饱和蜜桃色腮红，保留真实毛孔")
