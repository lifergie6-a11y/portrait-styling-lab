import json
from pathlib import Path

from PIL import Image

from portrait_styling.evaluator import cosine_similarity, select_best_candidate
from portrait_styling.report import generate_html_report
from portrait_styling.schema import CandidateEval, GeneratedCandidate


def test_select_best_candidate_enforces_id_gate():
    c_fail = GeneratedCandidate(
        model_alias="m1",
        model_id="m1",
        seed=1,
        image_path="a.png",
        latency_sec=3.0,
        prompt_used="p",
        eval_result=CandidateEval(id_similarity=0.50, id_passed=False, composite_score=90.0),
    )
    c_pass = GeneratedCandidate(
        model_alias="m1",
        model_id="m1",
        seed=2,
        image_path="b.png",
        latency_sec=3.1,
        prompt_used="p",
        eval_result=CandidateEval(id_similarity=0.78, id_passed=True, composite_score=78.0),
    )
    best = select_best_candidate([c_fail, c_pass])
    assert best is not None
    assert best.seed == 2
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6


def test_generate_html_report(tmp_path: Path):
    orig = tmp_path / "orig.png"
    gen = tmp_path / "gen.png"
    Image.new("RGB", (240, 320), (180, 140, 120)).save(orig)
    Image.new("RGB", (240, 320), (190, 150, 130)).save(gen)

    results = {
        "meta": {"generated_at": "2026-11-19T16:00:00Z", "id_threshold": 0.65},
        "models": ["firered-1.1", "qwen-2.1"],
        "records": [
            {
                "sample_id": "demo_001",
                "source_image": str(orig),
                "preprocessed_image": str(orig),
                "diagnosis": {"face_shape": "方圆脸", "skin_tone": "秋季暖皮"},
                "translated_prompt": {"prompt_zh": "测试提示词"},
                "model_runs": {
                    "firered-1.1": {
                        "best_candidate": {
                            "image_path": str(gen),
                            "seed": 42,
                            "latency_sec": 4.5,
                            "eval_result": {"id_similarity": 0.81, "id_passed": True, "vlm_score": 8.7},
                            "error": None,
                        },
                        "candidates": [],
                    },
                    "qwen-2.1": {
                        "best_candidate": {
                            "image_path": str(gen),
                            "seed": 42,
                            "latency_sec": 6.2,
                            "eval_result": {"id_similarity": 0.76, "id_passed": True, "vlm_score": 8.4},
                            "error": None,
                        },
                        "candidates": [],
                    },
                },
            }
        ],
    }

    res_json = tmp_path / "benchmark_results.json"
    res_json.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    out_html = tmp_path / "report.html"
    generate_html_report(res_json, out_html, embed_images=True)

    content = out_html.read_text(encoding="utf-8")
    assert "盲测模式" in content
    assert "firered-1.1" in content
    assert "data:image/jpeg;base64," in content
