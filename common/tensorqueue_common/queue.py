"""
Redis-backed job queue with:
  - a pending LIST for ready-to-run jobs (BRPOP for blocking consumption)
  - a delayed ZSET (score = ready-at epoch) for exponential-backoff retries
  - a processing ZSET (score = visibility-timeout deadline) so a crashed
    worker's in-flight job gets reclaimed instead of silently vanishing
  - a dead-letter LIST for jobs that exhausted their retry budget

This is intentionally built on primitives available in stock Redis (no
Streams/modules required) so it runs anywhere, including a single-pod
Redis Deployment in the cluster.
"""
from __future__ import annotations

import json
import random
import time

import redis

from .config import settings
from .models import Job, JobStatus, QueueStats


class QueueFull(Exception):
    pass


class RedisJobQueue:
    def __init__(self, client: redis.Redis | None = None):
        self.r = client or redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
        )

    # ---------- internal helpers ----------

    def _job_key(self, job_id: str) -> str:
        return f"{settings.job_hash_prefix}{job_id}"

    def _save_job(self, job: Job) -> None:
        job.touch()
        self.r.set(self._job_key(job.job_id), job.model_dump_json(), ex=settings.job_ttl_seconds)

    def _compute_backoff(self, retries: int) -> float:
        """Exponential backoff with full jitter, capped at max_backoff_seconds."""
        raw = settings.base_backoff_seconds * (2 ** retries)
        capped = min(raw, settings.max_backoff_seconds)
        jitter = random.uniform(0, settings.backoff_jitter_seconds)
        return capped + jitter

    # ---------- producer (API) ----------

    def enqueue(self, job: Job) -> Job:
        job.status = JobStatus.QUEUED
        self._save_job(job)
        self.r.lpush(settings.queue_pending, job.job_id)
        return job

    def get_job(self, job_id: str) -> Job | None:
        raw = self.r.get(self._job_key(job_id))
        if raw is None:
            return None
        return Job.model_validate_json(raw)

    # ---------- consumer (worker) ----------

    def dequeue(self, timeout: int | None = None) -> Job | None:
        """
        Blocking pop from the pending list. On success, immediately marks the
        job 'processing' and adds it to the processing ZSET with a visibility
        deadline so a crashed worker's job can be reclaimed later.
        """
        timeout = timeout if timeout is not None else settings.worker_poll_timeout_seconds
        popped = self.r.brpop(settings.queue_pending, timeout=timeout)
        if popped is None:
            return None
        _, job_id = popped

        job = self.get_job(job_id)
        if job is None:
            # Job hash expired/missing (TTL race) — nothing to do.
            return None

        job.status = JobStatus.PROCESSING
        self._save_job(job)
        deadline = time.time() + settings.processing_timeout_seconds
        self.r.zadd(settings.queue_processing, {job_id: deadline})
        return job

    def ack(self, job: Job, detections, latency_ms: float) -> None:
        """Mark a job successfully completed."""
        job.status = JobStatus.COMPLETED
        job.detections = detections
        job.latency_ms = latency_ms
        job.error = None
        self._save_job(job)
        self.r.zrem(settings.queue_processing, job.job_id)

    def fail(self, job: Job, error: str) -> None:
        """
        Handle a failed attempt: schedule an exponential-backoff retry, or
        move to the dead-letter queue once max_retries is exhausted.
        """
        self.r.zrem(settings.queue_processing, job.job_id)
        job.retries += 1
        job.error = error

        if job.retries > job.max_retries:
            job.status = JobStatus.DEAD_LETTER
            self._save_job(job)
            self.r.lpush(settings.queue_dlq, job.job_id)
            return

        delay = self._compute_backoff(job.retries)
        job.status = JobStatus.FAILED
        job.next_attempt_at = time.time() + delay
        self._save_job(job)
        self.r.zadd(settings.queue_delayed, {job.job_id: job.next_attempt_at})

    # ---------- background maintenance (run by worker or a sidecar) ----------

    def requeue_due_delayed(self, batch_size: int = 100) -> int:
        """Move delayed jobs whose backoff has elapsed back onto the pending list."""
        now = time.time()
        due = self.r.zrangebyscore(settings.queue_delayed, min=0, max=now, start=0, num=batch_size)
        if not due:
            return 0
        pipe = self.r.pipeline()
        for job_id in due:
            pipe.zrem(settings.queue_delayed, job_id)
            pipe.lpush(settings.queue_pending, job_id)
        pipe.execute()
        return len(due)

    def reclaim_stale_processing(self, batch_size: int = 100) -> int:
        """
        Safety net for crashed/killed workers: if a job has sat in the
        processing ZSET past its visibility deadline, treat it as a failed
        attempt and route it back through the normal backoff/DLQ logic.
        """
        now = time.time()
        stale = self.r.zrangebyscore(settings.queue_processing, min=0, max=now, start=0, num=batch_size)
        for job_id in stale:
            job = self.get_job(job_id)
            self.r.zrem(settings.queue_processing, job_id)
            if job is None:
                continue
            self.fail(job, error="worker_timeout: visibility deadline exceeded")
        return len(stale)

    # ---------- observability ----------

    def stats(self) -> QueueStats:
        return QueueStats(
            pending=self.r.llen(settings.queue_pending),
            delayed=self.r.zcard(settings.queue_delayed),
            processing=self.r.zcard(settings.queue_processing),
            dead_letter=self.r.llen(settings.queue_dlq),
        )

    def dlq_peek(self, count: int = 20) -> list[Job]:
        ids = self.r.lrange(settings.queue_dlq, 0, count - 1)
        jobs = [self.get_job(i) for i in ids]
        return [j for j in jobs if j is not None]

    def dlq_requeue(self, job_id: str) -> bool:
        """Manually reinstate a dead-lettered job for another attempt (ops action)."""
        job = self.get_job(job_id)
        if job is None:
            return False
        removed = self.r.lrem(settings.queue_dlq, 1, job_id)
        if not removed:
            return False
        job.retries = 0
        job.error = None
        job.status = JobStatus.QUEUED
        self._save_job(job)
        self.r.lpush(settings.queue_pending, job.job_id)
        return True
