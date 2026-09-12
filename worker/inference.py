"""
Thin wrapper around an Ultralytics YOLO model. Kept separate from the
queue-consumption loop so the model lifecycle (load-once, warm, reuse) is
easy to reason about and easy to swap out (e.g. for a different YOLO
checkpoint or a TensorRT-compiled engine) without touching worker.py.
"""
from __future__ import annotations

import logging
import time

import torch
from ultralytics import YOLO

from tensorqueue_common import Detection, settings

logger = logging.getLogger("tensorqueue.worker.inference")


class Detector:
    def __init__(self):
        device = settings.model_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

        logger.info("loading model weights=%s device=%s", settings.model_weights, device)
        self.device = device
        self.model = YOLO(settings.model_weights)
        self.model.to(device)

        # Warm-up pass so the first *real* job isn't the one eating JIT /
        # CUDA-context / kernel-autotune overhead — this is what keeps p99
        # tight instead of just p50.
        self._warmup()

    def _warmup(self) -> None:
        import numpy as np

        dummy = np.zeros((640, 640, 3), dtype="uint8")
        t0 = time.perf_counter()
        self.model.predict(dummy, device=self.device, verbose=False)
        logger.info("warmup inference took %.1fms", (time.perf_counter() - t0) * 1000)

    def predict(self, image_path: str) -> tuple[list[Detection], float]:
        t0 = time.perf_counter()
        results = self.model.predict(
            source=image_path,
            device=self.device,
            conf=settings.confidence_threshold,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        detections: list[Detection] = []
        if results:
            result = results[0]
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(Detection(label=names[cls_id], confidence=conf, box_xyxy=xyxy))

        return detections, latency_ms
