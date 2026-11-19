"""Unified CLI (`pstyle`) for single-image smoke tests, multi-model A/B benchmarks, and reporting."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from .evaluator import AutoScorer, select_best_candidate
from .generators import GenerationRequest, create_generator, load_model_registry
from .preprocess import FaceAnchorEngine
from .prompt_translator import translate_rule_based, translate_with_llm
from .report import generate_html_report
from .schema import (
    BenchmarkRecord,
    DiagnosisInput,
    GeneratedCandidate,
    SampleModelRun,
    resolve_optional_path,
)

logger = logging.getLogger("portrait_styling")


def _parse_size(size_str: str) -> tuple[int, int]:
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size '{size_str}'; expected WIDTHxHEIGHT (e.g. 768x1024).")
    return int(parts[0]), int(parts[1])


def _collect_reference_paths(diag: DiagnosisInput, base_dir: Path | None) -> list[Path]:
    refs: list[Path] = []
    for key in ("hair", "makeup"):
        p = resolve_optional_path(diag.reference_images.get(key), base_dir=base_dir)
        if p and p.exists():
            refs.append(p)
    return refs


def run_benchmark(
    samples: list[dict[str, Any]],
    model_aliases: list[str],
    output_dir: Path,
    base_dir: Path | None = None,
    num_seeds: int = 2,
    base_seed: int = 42,
    target_size: tuple[int, int] = (768, 1024),
    id_threshold: float = 0.65,
    use_llm_translator: bool = False,
    quantization: str = "auto",
    cpu_offload: str = "auto",
    cache_dir: str | None = None,
    colabllm_instance: str | None = None,
    models_yaml: str | None = None,
    steps_override: int | None = None,
) -> Path:
    """Run 4-stage pipeline across N samples x M models (model-major loop to minimize GPU reloads)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_model_registry(models_yaml)

    face_engine = FaceAnchorEngine(use_gpu=True)
    scorer = AutoScorer(face_engine=face_engine, id_threshold=id_threshold)

    # Stage 1 & Stage 2: Preprocess all portraits and translate all diagnosis prompts first
    prepared: list[dict[str, Any]] = []
    for idx, item in enumerate(samples):
        diag = DiagnosisInput.from_dict(item.get("diagnosis") or item)
        sample_id = str(item.get("sample_id") or diag.sample_id or f"sample_{idx + 1:03d}")
        diag.sample_id = sample_id

        img_path = resolve_optional_path(item.get("image"), base_dir=base_dir)
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(f"Sample '{sample_id}' image not found: {item.get('image')}")

        aligned_path = output_dir / "preprocessed" / f"{sample_id}.png"
        prep = face_engine.preprocess(img_path, aligned_path, target_size=target_size)

        translated = (
            translate_with_llm(diag) if use_llm_translator else translate_rule_based(diag)
        )
        refs = _collect_reference_paths(diag, base_dir=base_dir)

        prepared.append(
            {
                "sample_id": sample_id,
                "diag": diag,
                "prep": prep,
                "translated": translated,
                "refs": refs,
                "record": BenchmarkRecord(
                    sample_id=sample_id,
                    source_image=str(img_path),
                    preprocessed_image=prep.cropped_path,
                    diagnosis=diag.to_dict(),
                    translated_prompt={
                        "prompt_zh": translated.prompt_zh,
                        "prompt_en": translated.prompt_en,
                        "negative_prompt": translated.negative_prompt,
                        "checklist": translated.checklist,
                        "translator_mode": translated.translator_mode,
                    },
                    model_runs={},
                ),
            }
        )

    # Stage 3 & Stage 4: Run one model across all samples, then unload cleanly before the next model
    seeds = [base_seed + i * 101 for i in range(num_seeds)]
    for alias in model_aliases:
        logger.info("=== Running model '%s' on %d sample(s) ===", alias, len(prepared))
        editor = create_generator(
            model_alias=alias,
            registry=registry,
            quantization=quantization,
            cpu_offload=cpu_offload,
            cache_dir=cache_dir,
            colabllm_instance=colabllm_instance,
        )
        safe_alias = alias.replace(":", "_").replace("/", "_")

        try:
            for entry in prepared:
                sample_id = entry["sample_id"]
                prep = entry["prep"]
                translated = entry["translated"]
                prompt_text = translated.select_prompt(editor.preferred_language)
                candidates: list[GeneratedCandidate] = []

                for seed in seeds:
                    cand_path = output_dir / "generated" / safe_alias / f"{sample_id}_seed{seed}.png"
                    req = GenerationRequest(
                        portrait_path=Path(prep.cropped_path),
                        prompt=prompt_text,
                        output_path=cand_path,
                        seed=seed,
                        negative_prompt=translated.negative_prompt,
                        reference_paths=entry["refs"],
                        width=target_size[0],
                        height=target_size[1],
                        steps=steps_override,
                    )
                    try:
                        out = editor.edit(req)
                        ev = scorer.evaluate_candidate(
                            original_path=prep.cropped_path,
                            source_embedding=prep.embedding,
                            candidate_path=out.image_path,
                            checklist=translated.checklist,
                        )
                        candidates.append(
                            GeneratedCandidate(
                                model_alias=alias,
                                model_id=editor.model_id,
                                seed=seed,
                                image_path=str(out.image_path),
                                latency_sec=out.latency_sec,
                                prompt_used=out.prompt_used,
                                eval_result=ev,
                            )
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed sample=%s model=%s seed=%d: %s", sample_id, alias, seed, exc
                        )
                        candidates.append(
                            GeneratedCandidate(
                                model_alias=alias,
                                model_id=editor.model_id,
                                seed=seed,
                                image_path="",
                                latency_sec=0.0,
                                prompt_used=prompt_text,
                                error=str(exc),
                            )
                        )

                best = select_best_candidate(candidates)
                run_obj = SampleModelRun(
                    sample_id=sample_id,
                    model_alias=alias,
                    model_id=editor.model_id,
                    best_candidate=best,
                    candidates=candidates,
                )
                entry["record"].model_runs[alias] = run_obj.to_dict()
        finally:
            editor.unload()

    results_payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "models": model_aliases,
            "num_seeds": num_seeds,
            "target_size": f"{target_size[0]}x{target_size[1]}",
            "id_threshold": id_threshold,
            "detector_backend": face_engine.backend_name,
        },
        "models": model_aliases,
        "records": [entry["record"].to_dict() for entry in prepared],
    }

    results_json = output_dir / "benchmark_results.json"
    results_json.write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_html = output_dir / "report.html"
    generate_html_report(results_json, report_html, embed_images=True)
    logger.info("Benchmark complete! Results: %s | Dashboard: %s", results_json, report_html)
    return report_html


def _cmd_doctor(_args: argparse.Namespace) -> int:
    print("=== Portrait Styling Lab Doctor ===")
    checks = []
    for mod in ("PIL", "yaml", "numpy", "requests", "insightface", "mediapipe", "torch", "diffusers", "bitsandbytes"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            checks.append((mod, True, ver))
        except Exception:
            checks.append((mod, False, "not installed"))

    for name, ok, info in checks:
        mark = "OK" if ok else "--"
        print(f"  [{mark}] {name:<14} {info}")

    colabllm_bin = shutil.which("colabllm")
    print(f"  [{'OK' if colabllm_bin else '--'}] {'colabllm CLI':<14} {colabllm_bin or 'not found'}")

    reg = load_model_registry()
    print(f"\nRegistered SFW models ({len(reg)}):")
    for alias, cfg in reg.items():
        print(f"  - {alias:<22} ({cfg.get('backend')}) -> {cfg.get('model_id')} [{cfg.get('license')}]")
    return 0


def _cmd_run_one(args: argparse.Namespace) -> int:
    diag_data = json.loads(Path(args.diagnosis).read_text(encoding="utf-8"))
    sample = {
        "sample_id": diag_data.get("sample_id", Path(args.image).stem),
        "image": str(Path(args.image).resolve()),
        "diagnosis": diag_data,
    }
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    report_path = run_benchmark(
        samples=[sample],
        model_aliases=models,
        output_dir=Path(args.output_dir),
        base_dir=Path(args.diagnosis).resolve().parent,
        num_seeds=args.num_seeds,
        base_seed=args.seed,
        target_size=_parse_size(args.size),
        id_threshold=args.id_threshold,
        use_llm_translator=args.use_llm_translator,
        quantization=args.quantization,
        cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
        colabllm_instance=args.colabllm_instance,
        models_yaml=args.models_yaml,
        steps_override=args.steps,
    )
    print(f"Done. Open dashboard: {report_path}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", manifest) if isinstance(manifest, dict) else manifest
    if args.limit:
        samples = samples[: args.limit]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    report_path = run_benchmark(
        samples=samples,
        model_aliases=models,
        output_dir=Path(args.output_dir),
        base_dir=manifest_path.parent,
        num_seeds=args.num_seeds,
        base_seed=args.seed,
        target_size=_parse_size(args.size),
        id_threshold=args.id_threshold,
        use_llm_translator=args.use_llm_translator,
        quantization=args.quantization,
        cpu_offload=args.cpu_offload,
        cache_dir=args.cache_dir,
        colabllm_instance=args.colabllm_instance,
        models_yaml=args.models_yaml,
        steps_override=args.steps,
    )
    print(f"Benchmark finished. Open HTML A/B report: {report_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    out = generate_html_report(
        results_json_path=args.results,
        output_html_path=args.output,
        embed_images=not args.no_embed,
        title=args.title,
    )
    print(f"Report written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pstyle",
        description="SFW Portrait Makeup & Hairstyle Editing Benchmark CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Check installed backends, GPU/eval libraries, and models")
    p_doc.set_defaults(func=_cmd_doctor)

    def _add_common_run_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", required=True, help="Directory for outputs and report.html")
        p.add_argument("--num-seeds", type=int, default=2, help="Best-of-N seeds per sample (default: 2)")
        p.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
        p.add_argument("--size", default="768x1024", help="Aligned portrait size WxH (default: 768x1024)")
        p.add_argument("--id-threshold", type=float, default=0.65, help="ArcFace ID cosine threshold")
        p.add_argument("--use-llm-translator", action="store_true", help="Use STYLING_LLM_* API for prompt translation")
        p.add_argument("--quantization", choices=["auto", "none", "nf4"], default="auto")
        p.add_argument("--cpu-offload", choices=["auto", "true", "false"], default="auto")
        p.add_argument("--cache-dir", default=None, help="Hugging Face cache directory (e.g. on Google Drive)")
        p.add_argument("--colabllm-instance", default=None, help="Optional `colabllm -i INSTANCE` ID")
        p.add_argument("--models-yaml", default=None, help="Custom models.yaml registry path")
        p.add_argument("--steps", type=int, default=None, help="Override inference steps")

    p_one = sub.add_parser("run-one", help="Smoke-test one portrait against one or more models")
    p_one.add_argument("--image", required=True, help="Input portrait image path")
    p_one.add_argument("--diagnosis", required=True, help="Diagnosis JSON file path")
    p_one.add_argument("--model", required=True, help="Model alias or comma-separated aliases")
    _add_common_run_flags(p_one)
    p_one.set_defaults(func=_cmd_run_one)

    p_bench = sub.add_parser("benchmark", help="Run A/B benchmark across a golden dataset manifest")
    p_bench.add_argument("--manifest", required=True, help="Path to dataset manifest.json")
    p_bench.add_argument("--models", required=True, help="Comma-separated model aliases (e.g. firered-1.1,qwen-2.1)")
    p_bench.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick dry runs")
    _add_common_run_flags(p_bench)
    p_bench.set_defaults(func=_cmd_benchmark)

    p_rep = sub.add_parser("report", help="Regenerate HTML A/B dashboard from benchmark_results.json")
    p_rep.add_argument("--results", required=True, help="Path to benchmark_results.json")
    p_rep.add_argument("--output", required=True, help="Output HTML path")
    p_rep.add_argument("--no-embed", action="store_true", help="Link image paths instead of embedding base64")
    p_rep.add_argument("--title", default="Portrait Styling Lab — Model A/B Comparison")
    p_rep.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
