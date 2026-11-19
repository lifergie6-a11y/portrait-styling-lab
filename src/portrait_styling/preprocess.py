"""Stage 1: Portrait-aware preprocessing, headroom framing, and ArcFace ID anchoring.

Strictly uses recognition/detection models (InsightFace buffalo_l or MediaPipe) for
framing and identity preservation scoring — never uses any face-swap models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .schema import PreprocessedPortrait

logger = logging.getLogger(__name__)


def compute_portrait_crop_box(
    img_w: int,
    img_h: int,
    face_bbox: tuple[float, float, float, float] | None,
    target_aspect: float = 3.0 / 4.0,
    top_headroom_ratio: float = 0.85,
    bottom_shoulder_ratio: float = 1.45,
) -> tuple[int, int, int, int]:
    """Compute a portrait crop box (left, top, right, bottom) with generous headroom & shoulders.

    Why generous headroom matters for hairstyle editing:
    Tight face crops clip the crown and shoulders, preventing editing models from
    generating voluminous crowns, bangs, or collarbone-length/long hair naturally.
    """
    if face_bbox is None:
        # Center-top heuristic fallback when no face detector is available
        cur_aspect = img_w / max(img_h, 1)
        if cur_aspect > target_aspect:
            crop_h = img_h
            crop_w = int(round(crop_h * target_aspect))
            left = max(0, (img_w - crop_w) // 2)
            top = 0
        else:
            crop_w = img_w
            crop_h = int(round(crop_w / target_aspect))
            left = 0
            top = max(0, int((img_h - crop_h) * 0.2))
        return (left, top, min(img_w, left + crop_w), min(img_h, top + crop_h))

    x1, y1, x2, y2 = face_bbox
    fw = max(x2 - x1, 1.0)
    fh = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0

    desired_top = y1 - fh * top_headroom_ratio
    desired_bottom = y2 + fh * bottom_shoulder_ratio
    desired_h = max(desired_bottom - desired_top, fh * 2.6)
    desired_w = desired_h * target_aspect

    # Clamp crop dimensions to image bounds while preserving target aspect ratio
    if desired_w > img_w:
        desired_w = float(img_w)
        desired_h = desired_w / target_aspect
    if desired_h > img_h:
        desired_h = float(img_h)
        desired_w = desired_h * target_aspect

    # Center horizontally around face center; place face upper-mid vertically (~36% from top)
    left = cx - desired_w / 2.0
    face_cy = (y1 + y2) / 2.0
    top = face_cy - desired_h * 0.38

    # Shift inside image boundaries
    left = max(0.0, min(left, img_w - desired_w))
    top = max(0.0, min(top, img_h - desired_h))

    return (
        int(round(left)),
        int(round(top)),
        int(round(min(img_w, left + desired_w))),
        int(round(min(img_h, top + desired_h))),
    )


class FaceAnchorEngine:
    """Lazy-loaded face detector & ArcFace embedding extractor.

    Priority order:
    1. InsightFace (`buffalo_l`): provides face bbox + 512-d ArcFace L2-normalized embedding.
    2. MediaPipe FaceDetection: provides face bbox only (no ID embedding).
    3. Fallback heuristic crop: zero extra dependencies.
    """

    def __init__(self, det_size: tuple[int, int] = (640, 640), use_gpu: bool = True) -> None:
        self.det_size = det_size
        self.use_gpu = use_gpu
        self._backend: str | None = None
        self._app: Any = None

    def _init_backend(self) -> str:
        if self._backend is not None:
            return self._backend

        # Try InsightFace first (gives both framing bbox AND ArcFace ID embedding)
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.use_gpu
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if self.use_gpu else -1, det_size=self.det_size)
            self._app = app
            self._backend = "insightface"
            logger.info("Initialized FaceAnchorEngine with InsightFace buffalo_l")
            return self._backend
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("InsightFace unavailable (%s), trying MediaPipe", exc)

        # Fallback to MediaPipe for framing only
        try:
            import mediapipe as mp  # type: ignore

            self._app = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self._backend = "mediapipe"
            logger.info("Initialized FaceAnchorEngine with MediaPipe (framing only)")
            return self._backend
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("MediaPipe unavailable (%s), using heuristic framing", exc)

        self._backend = "heuristic"
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._init_backend()

    def analyze_image(self, image: Image.Image) -> tuple[list[float] | None, list[float] | None]:
        """Return (primary_face_bbox [x1, y1, x2, y2], normalized_embedding) for a PIL RGB image."""
        backend = self._init_backend()
        rgb = np.array(image.convert("RGB"))

        if backend == "insightface":
            # InsightFace expects BGR uint8
            bgr = rgb[:, :, ::-1].copy()
            faces = self._app.get(bgr)
            if not faces:
                return None, None
            # Choose largest detected face as primary subject
            primary = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            bbox = [float(v) for v in primary.bbox.tolist()]
            emb = getattr(primary, "normed_embedding", None)
            if emb is None and getattr(primary, "embedding", None) is not None:
                raw = np.asarray(primary.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(raw))
                emb = raw / max(norm, 1e-12)
            emb_list = [float(v) for v in emb.tolist()] if emb is not None else None
            return bbox, emb_list

        if backend == "mediapipe":
            results = self._app.process(rgb)
            if not results.detections:
                return None, None
            h, w = rgb.shape[:2]
            det = max(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height,
            )
            rb = det.location_data.relative_bounding_box
            x1 = max(0.0, rb.xmin * w)
            y1 = max(0.0, rb.ymin * h)
            x2 = min(float(w), (rb.xmin + rb.width) * w)
            y2 = min(float(h), (rb.ymin + rb.height) * h)
            return [x1, y1, x2, y2], None

        return None, None

    def preprocess(
        self,
        source_path: str | Path,
        output_path: str | Path,
        target_size: tuple[int, int] = (768, 1024),
    ) -> PreprocessedPortrait:
        """Align EXIF orientation, portrait-crop with headroom, resize, and extract ID embedding."""
        src = Path(source_path)
        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        bbox, _ = self.analyze_image(img)

        target_w, target_h = target_size
        crop_box = compute_portrait_crop_box(
            img_w=img.width,
            img_h=img.height,
            face_bbox=tuple(bbox) if bbox is not None else None,
            target_aspect=target_w / target_h,
        )
        cropped = img.crop(crop_box).resize((target_w, target_h), Image.Resampling.LANCZOS)
        cropped.save(dst, format="PNG")

        # Extract final reference embedding on the aligned cropped portrait
        aligned_bbox, aligned_emb = self.analyze_image(cropped)
        return PreprocessedPortrait(
            source_path=str(src),
            cropped_path=str(dst),
            width=target_w,
            height=target_h,
            face_detected=aligned_bbox is not None or bbox is not None,
            face_bbox=aligned_bbox or bbox,
            embedding=aligned_emb,
            detector_backend=self.backend_name,
        )
