"""
TensorQueue API Gateway.

Client-facing FastAPI service. Fully decoupled from the compute layer: it
only ever talks to Redis (the queue) and a shared image volume — it never
imports torch or touches a model. That decoupling is what lets the worker
fleet scale/restart/crash independently of the API's request-serving SLO.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from tensorqueue_common import Job, JobStatus, JobStatusResponse, JobSubmitResponse, MetricsResponse, settings
from tensorqueue_common.queue import RedisJobQueue
from tensorqueue_common.metrics import read_percentiles

from .storage import save_upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tensorqueue.api")

app = FastAPI(title="TensorQueue API Gateway", version="0.1.0")
queue = RedisJobQueue()


@app.get("/healthz")
def healthz():
    """Liveness/readiness probe. Verifies Redis connectivity so k8s can
    pull a broken pod out of rotation instead of routing traffic to it."""
    try:
        queue.r.ping()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"redis unavailable: {exc}") from exc
    return {"status": "ok"}


@app.post("/v1/jobs", response_model=JobSubmitResponse, status_code=202)
async def submit_job(file: UploadFile):
    """
    Accept an image, persist it, and enqueue an inference job. Returns
    immediately (202 Accepted) with a job_id for polling — the client never
    blocks on inference, which is what lets a single gateway pod front a
    much larger, independently-scaled worker fleet.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/bmp", "image/webp"):
        raise HTTPException(status_code=415, detail=f"unsupported content type: {file.content_type}")

    image_path = await save_upload(file)
    job = Job(image_path=image_path)
    queue.enqueue(job)
    logger.info("enqueued job=%s image=%s", job.job_id, image_path)

    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status,
        poll_url=f"/v1/jobs/{job.job_id}",
    )


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found (unknown id or TTL expired)")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        retries=job.retries,
        detections=job.detections,
        latency_ms=job.latency_ms,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@app.get("/v1/metrics", response_model=MetricsResponse)
def metrics():
    """JSON metrics (p50/p95/p99 + queue depth) for quick inspection without
    standing up Prometheus/Grafana. Worker pods also expose a proper
    Prometheus histogram on :METRICS_PORT/metrics for cluster monitoring
    and HPA-driven autoscaling."""
    pct = read_percentiles(queue.r)
    return MetricsResponse(
        p50_latency_ms=pct["p50_latency_ms"],
        p95_latency_ms=pct["p95_latency_ms"],
        p99_latency_ms=pct["p99_latency_ms"],
        sample_count=pct["sample_count"],
        queue=queue.stats(),
    )


@app.get("/v1/dlq")
def list_dead_letters(count: int = 20):
    """Inspect the dead-letter queue — jobs that exhausted MAX_RETRIES."""
    jobs = queue.dlq_peek(count)
    return JSONResponse([j.model_dump(mode="json") for j in jobs])


@app.post("/v1/dlq/{job_id}/requeue")
def requeue_dead_letter(job_id: str):
    """Ops escape hatch: manually resubmit a dead-lettered job (e.g. after
    fixing a bad model deploy) without the client having to re-upload."""
    ok = queue.dlq_requeue(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found in dead-letter queue")
    return {"job_id": job_id, "status": JobStatus.QUEUED}
