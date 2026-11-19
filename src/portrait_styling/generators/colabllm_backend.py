"""Remote Colab acceleration backend via installed `colabllm` CLI.

Allows running A/B portrait benchmarks directly from a local Mac/Linux machine while
offloading heavy GPU inference to an active `colabllm qwen-image` or `colabllm flux-klein`
Google Colab runtime.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import Any

from .base import BaseImageEditor, GenerationOutput, GenerationRequest

logger = logging.getLogger(__name__)


class ColabLLMImageEditor(BaseImageEditor):
    """Wraps `colabllm <service> edit` for Qwen-Image-2.1 and FLUX 3 Image Klein."""

    SUPPORTED_SERVICES = ("qwen-image", "flux-klein")

    def __init__(
        self,
        alias: str,
        config: dict[str, Any],
        instance: str | None = None,
        timeout: int = 300,
    ) -> None:
        super().__init__(alias, config)
        self.service = str(config.get("service", "")).strip()
        if self.service not in self.SUPPORTED_SERVICES:
            raise ValueError(
                f"Unsupported colabllm service '{self.service}'. Supported: {self.SUPPORTED_SERVICES}"
            )
        self.defaults: dict[str, Any] = dict(config.get("defaults") or {})
        self.instance = instance
        self.timeout = timeout

    def _find_binary(self) -> str:
        binary = shutil.which("colabllm")
        if not binary:
            raise RuntimeError(
                "`colabllm` CLI not found in PATH. Install colabllm or use a diffusers backend "
                "(e.g., `firered-1.1` / `qwen-2.1`) inside Google Colab."
            )
        return binary

    def build_command(self, request: GenerationRequest) -> list[str]:
        binary = self._find_binary()
        cmd = [binary, self.service]
        if self.instance:
            cmd.extend(["-i", self.instance])
        cmd.append("edit")

        all_images = [str(request.portrait_path), *[str(p) for p in request.reference_paths]]
        cmd.extend(["--image", *all_images])

        size = f"{request.width}x{request.height}" if request.width and request.height else str(
            self.defaults.get("size", "768x1024")
        )
        cmd.extend(["--size", size])
        cmd.extend(["--seed", str(request.seed)])

        steps = request.steps or self.defaults.get("steps")
        if steps:
            cmd.extend(["--steps", str(steps)])

        if self.service == "qwen-image":
            neg = request.negative_prompt.strip() or str(self.defaults.get("negative_prompt", "")).strip()
            if neg:
                cmd.extend(["--negative-prompt", neg])
            if "true_cfg_scale" in self.defaults and not neg:
                cmd.extend(["--true-cfg-scale", str(self.defaults["true_cfg_scale"])])

        cmd.extend(["--timeout", str(self.timeout)])
        cmd.extend(["-o", str(request.output_path)])
        cmd.extend(["-p", request.prompt])
        return cmd

    def edit(self, request: GenerationRequest) -> GenerationOutput:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(request)
        logger.info("Executing colabllm edit: service=%s seed=%d", self.service, request.seed)

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout + 30,
        )
        latency = time.perf_counter() - t0

        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"colabllm {self.service} edit failed (exit={proc.returncode}): {err_msg}\n"
                f"Tip: ensure the runtime is up via `colabllm {self.service} status --json`."
            )

        if not request.output_path.exists():
            raise RuntimeError(
                f"colabllm {self.service} exited 0 but output file was not written: {request.output_path}"
            )

        return GenerationOutput(
            image_path=request.output_path,
            latency_sec=round(latency, 3),
            seed=request.seed,
            model_alias=self.alias,
            model_id=self.model_id,
            prompt_used=request.prompt,
        )
