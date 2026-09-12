"""
Exercises the core fault-tolerance logic — enqueue/dequeue, exponential
backoff scheduling, stale-processing reclaim, and DLQ routing — against an
in-memory fake Redis so it runs in CI with zero infra.
"""
import time

import fakeredis
import pytest

from tensorqueue_common.config import settings
from tensorqueue_common.models import Job, JobStatus
from tensorqueue_common.queue import RedisJobQueue


@pytest.fixture
def queue():
    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisJobQueue(client=client)


def make_job(**overrides) -> Job:
    defaults = dict(image_path="/data/images/test.jpg")
    defaults.update(overrides)
    return Job(**defaults)


def test_enqueue_dequeue_roundtrip(queue):
    job = queue.enqueue(make_job())
    assert queue.stats().pending == 1

    popped = queue.dequeue(timeout=1)
    assert popped is not None
    assert popped.job_id == job.job_id
    assert popped.status == JobStatus.PROCESSING
    assert queue.stats().processing == 1
    assert queue.stats().pending == 0


def test_ack_marks_completed_and_clears_processing(queue):
    job = queue.enqueue(make_job())
    popped = queue.dequeue(timeout=1)
    queue.ack(popped, detections=[], latency_ms=42.0)

    stored = queue.get_job(job.job_id)
    assert stored.status == JobStatus.COMPLETED
    assert stored.latency_ms == 42.0
    assert queue.stats().processing == 0


def test_fail_schedules_exponential_backoff(queue):
    job = queue.enqueue(make_job())
    popped = queue.dequeue(timeout=1)

    queue.fail(popped, error="model exploded")

    stored = queue.get_job(job.job_id)
    assert stored.status == JobStatus.FAILED
    assert stored.retries == 1
    assert stored.next_attempt_at is not None

    # First backoff should be roughly base * 2^1 (+ jitter), not immediate
    expected_min = settings.base_backoff_seconds * 2
    assert stored.next_attempt_at >= time.time() + expected_min - 0.5
    assert queue.stats().delayed == 1
    assert queue.stats().pending == 0


def test_backoff_grows_exponentially_across_attempts(queue):
    job = queue.enqueue(make_job())
    delays = []
    for _ in range(3):
        popped = queue.dequeue(timeout=1)
        before = time.time()
        queue.fail(popped, error="still exploding")
        stored = queue.get_job(job.job_id)
        delays.append(stored.next_attempt_at - before)
        # force it due immediately so we can dequeue again next iteration
        queue.r.zadd(settings.queue_delayed, {job.job_id: 0})
        queue.requeue_due_delayed()

    # each successive backoff should be larger than the last
    assert delays[0] < delays[1] < delays[2]


def test_exhausted_retries_routes_to_dlq(queue):
    job = queue.enqueue(make_job(max_retries=2))

    for _ in range(3):  # 1 initial + 2 retries = 3 failing attempts total
        popped = queue.dequeue(timeout=1)
        assert popped is not None, "job should have been requeued for retry"
        queue.fail(popped, error="permanent failure")
        stored = queue.get_job(job.job_id)
        if stored.status == JobStatus.DEAD_LETTER:
            break  # exhausted — nothing left to force back onto pending
        queue.r.zadd(settings.queue_delayed, {job.job_id: 0})
        queue.requeue_due_delayed()

    stored = queue.get_job(job.job_id)
    assert stored.status == JobStatus.DEAD_LETTER
    assert queue.stats().dead_letter == 1
    assert queue.stats().pending == 0
    assert queue.stats().delayed == 0


def test_dlq_requeue_resets_and_reinstates_job(queue):
    job = queue.enqueue(make_job(max_retries=0))
    popped = queue.dequeue(timeout=1)
    queue.fail(popped, error="dead on arrival")
    assert queue.get_job(job.job_id).status == JobStatus.DEAD_LETTER

    ok = queue.dlq_requeue(job.job_id)
    assert ok is True

    stored = queue.get_job(job.job_id)
    assert stored.status == JobStatus.QUEUED
    assert stored.retries == 0
    assert queue.stats().pending == 1
    assert queue.stats().dead_letter == 0


def test_stale_processing_job_is_reclaimed_after_worker_crash(queue):
    job = queue.enqueue(make_job())
    queue.dequeue(timeout=1)  # simulates a worker picking it up, then dying

    # Force the visibility deadline into the past to simulate the timeout elapsing
    queue.r.zadd(settings.queue_processing, {job.job_id: time.time() - 1})

    reclaimed = queue.reclaim_stale_processing()
    assert reclaimed == 1

    stored = queue.get_job(job.job_id)
    assert stored.status == JobStatus.FAILED
    assert stored.retries == 1
    assert queue.stats().processing == 0


def test_requeue_due_delayed_only_moves_jobs_whose_time_has_come(queue):
    job = queue.enqueue(make_job())
    popped = queue.dequeue(timeout=1)
    queue.fail(popped, error="transient")  # scheduled well in the future

    moved = queue.requeue_due_delayed()
    assert moved == 0  # backoff hasn't elapsed yet
    assert queue.stats().pending == 0
    assert queue.stats().delayed == 1
