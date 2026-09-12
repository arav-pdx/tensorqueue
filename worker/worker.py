"""
TensorQueue inference worker.

Consumes jobs from Redis, runs YOLO object detection, and reports
success/failure back through the shared RedisJobQueue (which owns all
retry/backoff/DLQ semantics — the worker just calls .ack()/.fail()).

Fault tolerance model:
  - a background thread periodically promotes due delayed (backoff) jobs
    back onto the pending list
  - the same thread reclaims jobs whose visibility deadline expired,
    which is what happens if a worker pod is OOMKilled or evicted
    mid-inference — the job doesn't just vanish
  - SIGTERM triggers a graceful drain: stop pulling new jobs, finish the
    in-flight one, exit 0, so a rolling deploy / HPA scale-down doesn't
    silently drop work
"""
from __future__ import annotations

import logging
import signal
import threading
import time

from prometheus_client import start_http_server

from tensorqueue_common import JobStatus, settings
from tensorqueue_common.queue import RedisJobQueue
from tensorqueue_common.metrics import record_latency

from inference import Detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tensorqueue.worker")

_shutdown = threading.Event()


def _handle_sigterm(signum, frame):  # noqa: ARG001
    logger.info("received signal %s, draining in-flight job then exiting", signum)
    _shutdown.set()


def _maintenance_loop(queue: RedisJobQueue) -> None:
    """Runs in a background thread: promotes due retries, reclaims stale jobs."""
    while not _shutdown.is_set():
        try:
            requeued = queue.requeue_due_delayed()
            reclaimed = queue.reclaim_stale_processing()
            if requeued or reclaimed:
                logger.info("maintenance: requeued=%d reclaimed_stale=%d", requeued, reclaimed)
        except Exception:  # noqa: BLE001
            logger.exception("maintenance loop error")
        _shutdown.wait(settings.requeue_scan_interval_seconds)


def run() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    start_http_server(settings.metrics_port)
    logger.info("prometheus metrics on :%d/metrics", settings.metrics_port)

    queue = RedisJobQueue()
    detector = Detector()

    maintenance = threading.Thread(target=_maintenance_loop, args=(queue,), daemon=True)
    maintenance.start()

    logger.info("worker ready, polling %s", settings.queue_pending)

    while not _shutdown.is_set():
        job = queue.dequeue(timeout=settings.worker_poll_timeout_seconds)
        if job is None:
            continue  # poll timeout — loop back and check _shutdown

        logger.info("processing job=%s retries=%d", job.job_id, job.retries)
        try:
            detections, latency_ms = detector.predict(job.image_path)
            queue.ack(job, detections, latency_ms)
            record_latency(queue.r, latency_ms)
            logger.info(
                "completed job=%s detections=%d latency_ms=%.1f",
                job.job_id, len(detections), latency_ms,
            )
        except FileNotFoundError as exc:
            # Not transient — retrying won't help, but we still route it
            # through fail() so it lands in the DLQ after max_retries for
            # visibility rather than being silently dropped.
            logger.error("job=%s image missing: %s", job.job_id, exc)
            queue.fail(job, error=f"image_not_found: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("job=%s failed (attempt %d)", job.job_id, job.retries + 1)
            queue.fail(job, error=str(exc))

    logger.info("shutdown complete")


if __name__ == "__main__":
    run()
