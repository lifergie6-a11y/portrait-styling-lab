"""Standalone Hugging Face Diffusers backend for Google Colab / local GPUs.

Supports:
- FireRedTeam/FireRed-Image-Edit-1.1  (DiffusionPipeline / QwenImageEditPlusPipeline)
- Qwen/Qwen-Image-2.1 & Turbo         (QwenImage21Pipeline)
- Qwen/Qwen-Image-Edit-2511           (QwenImageEditPlusPipeline)

Includes automatic VRAM-aware loading:
- High VRAM (>= 48 GB, e.g., Colab G4 96GB / A100 80GB / H100): full BF16 directly on CUDA.
- Mid VRAM (20-47 GB, e.g., Colab L4 24GB / A100 40GB): CPU offload + optional 4-bit NF4 quantization.
"""

from __future__ import annotations

import gc
import inspect
import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .base import BaseImageEditor, GenerationOutput, GenerationRequest

logger = logging.getLogger(__name__)


class DiffusersImageEditor(BaseImageEditor):
    """Runs official SFW diffusers image-editing pipelines directly on GPU."""

    def __init__(
        self,
        alias: str,
        config: dict[str, Any],
        quantization: str = "auto",
        cpu_offload: str = "auto",
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(alias, config)
        self.pipeline_class_name = str(config.get("pipeline_class", "DiffusionPipeline"))
        self.defaults: dict[str, Any] = dict(config.get("defaults") or {})
        self.quantization = quantization
        self.cpu_offload = cpu_offload
        self.cache_dir = cache_dir
        self._pipe: Any = None

    def _detect_vram_gb(self) -> float:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)

    def _resolve_pipeline_class(self) -> Any:
        import diffusers  # type: ignore

        cls = getattr(diffusers, self.pipeline_class_name, None)
        if cls is None:
            if self.pipeline_class_name == "QwenImage21Pipeline":
                raise ImportError(
                    "QwenImage21Pipeline requires latest diffusers from git main. "
                    "Install with: pip install git+https://github.com/huggingface/diffusers.git"
                )
            logger.warning(
                "Pipeline class %s not found in diffusers; falling back to DiffusionPipeline.",
                self.pipeline_class_name,
            )
            return diffusers.DiffusionPipeline
        return cls

    def _ensure_loaded(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        import torch  # type: ignore

        vram_gb = self._detect_vram_gb()
        pipe_cls = self._resolve_pipeline_class()

        use_nf4 = self.quantization == "nf4" or (
            self.quantization == "auto" and 0 < vram_gb < 28.0
        )
        use_offload = self.cpu_offload == "true" or (
            self.cpu_offload == "auto" and 0 < vram_gb < 48.0
        )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        if use_nf4:
            try:
                from diffusers import BitsAndBytesConfig  # type: ignore

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info("Loading %s with 4-bit NF4 quantization (VRAM=%.1f GB)", self.model_id, vram_gb)
            except Exception as exc:
                logger.warning("4-bit NF4 quantization unavailable (%s); falling back to BF16+offload.", exc)
                use_offload = True

        logger.info(
            "Loading diffusers pipeline %s (%s) [VRAM=%.1f GB, offload=%s]",
            self.model_id,
            pipe_cls.__name__,
            vram_gb,
            use_offload,
        )
        pipe = pipe_cls.from_pretrained(self.model_id, **load_kwargs)

        if torch.cuda.is_available():
            if use_offload and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to("cuda")
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

        self._pipe = pipe
        return self._pipe

    def edit(self, request: GenerationRequest) -> GenerationOutput:
        import torch  # type: ignore

        pipe = self._ensure_loaded()
        request.output_path.parent.mkdir(parents=True, exist_ok=True)

        portrait = Image.open(request.portrait_path).convert("RGB")
        ref_images = [Image.open(p).convert("RGB") for p in request.reference_paths]
        image_input: Image.Image | list[Image.Image] = (
            [portrait, *ref_images] if ref_images else portrait
        )

        steps = request.steps or int(self.defaults.get("num_inference_steps", 40))
        neg_prompt = (
            request.negative_prompt
            if request.negative_prompt
            else str(self.defaults.get("negative_prompt", ""))
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(request.seed)

        # Dynamically filter kwargs against pipeline __call__ signature for cross-version safety
        sig_params = set(inspect.signature(pipe.__call__).parameters.keys())
        candidate_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "image": image_input,
            "num_inference_steps": steps,
            "generator": generator,
            "width": request.width,
            "height": request.height,
        }
        if neg_prompt and "negative_prompt" in sig_params:
            candidate_kwargs["negative_prompt"] = neg_prompt
        if "true_cfg_scale" in sig_params:
            # For Qwen-Image-2.1, only engage true_cfg_scale > 1.0 when negative_prompt is non-empty
            default_tcfg = float(self.defaults.get("true_cfg_scale", 1.0))
            if self.pipeline_class_name == "QwenImage21Pipeline" and not neg_prompt.strip():
                candidate_kwargs["true_cfg_scale"] = 1.0
            else:
                candidate_kwargs["true_cfg_scale"] = default_tcfg
        if "guidance_scale" in sig_params and "guidance_scale" in self.defaults:
            candidate_kwargs["guidance_scale"] = float(self.defaults["guidance_scale"])

        call_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig_params}

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = pipe(**call_kwargs)
        latency = time.perf_counter() - t0

        out_img = result.images[0]
        out_img.save(request.output_path, format="PNG")

        return GenerationOutput(
            image_path=request.output_path,
            latency_sec=round(latency, 3),
            seed=request.seed,
            model_alias=self.alias,
            model_id=self.model_id,
            prompt_used=request.prompt,
        )

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
