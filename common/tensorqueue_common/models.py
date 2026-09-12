from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .config import settings


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"          # transient failure, retry scheduled
    DEAD_LETTER = "dead_letter"  # exhausted retries


class Detection(BaseModel):
    label: str
    confidence: float
    box_xyxy: list[float] = Field(description="[x1, y1, x2, y2] in pixel coords")


class Job(BaseModel):
    """
    The canonical job record. Serialized as a JSON string and stored in a
    Redis hash field (tensorqueue:job:{id} -> {"data": <json>}) so both the
    API and worker share one source of truth for status.
    """
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    image_path: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    retries: int = 0
    max_retries: int = Field(default_factory=lambda: settings.max_retries)
    error: str | None = None
    detections: list[Detection] | None = None
    latency_ms: float | None = None
    next_attempt_at: float | None = None

    def touch(self) -> None:
        self.updated_at = time.time()


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    retries: int
    detections: list[Detection] | None = None
    latency_ms: float | None = None
    error: str | None = None
    created_at: float
    updated_at: float


class QueueStats(BaseModel):
    pending: int
    delayed: int
    processing: int
    dead_letter: int
    throughput_last_hour: int | None = None


class MetricsResponse(BaseModel):
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    sample_count: int
    queue: QueueStats
