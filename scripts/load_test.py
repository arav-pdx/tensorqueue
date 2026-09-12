#!/usr/bin/env python3
"""
Simple load generator against the API gateway.

Two modes:
  --mode steady   submits at a steady rate matching a target jobs/day figure
                   (default 1500/day ~= one job every 57.6s)
  --mode burst     fires `--count` jobs as fast as possible, then reports
                   p50/p95/p99 client-observed latency + polls until every
                   job reaches a terminal state, to sanity-check backpressure
                   and worker autoscaling under load

Usage:
  python scripts/load_test.py --mode burst --count 200 --image sample.jpg
  python scripts/load_test.py --mode steady --jobs-per-day 1500 --image sample.jpg
"""
from __future__ import annotations

import argparse
import statistics
import time

import requests


def submit_one(base_url: str, image_path: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    with open(image_path, "rb") as f:
        resp = requests.post(f"{base_url}/v1/jobs", files={"file": f})
    resp.raise_for_status()
    submit_latency = (time.perf_counter() - t0) * 1000
    return resp.json()["job_id"], submit_latency


def poll_until_terminal(base_url: str, job_id: str, timeout_s: float = 120) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.get(f"{base_url}/v1/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        if data["status"] in ("completed", "dead_letter"):
            return data
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not reach a terminal state in {timeout_s}s")


def run_burst(base_url: str, count: int, image_path: str) -> None:
    print(f"submitting {count} jobs as fast as possible...")
    submit_latencies = []
    job_ids = []
    for i in range(count):
        job_id, lat = submit_one(base_url, image_path)
        submit_latencies.append(lat)
        job_ids.append(job_id)
        if (i + 1) % 50 == 0:
            print(f"  submitted {i + 1}/{count}")

    print("waiting for all jobs to reach a terminal state...")
    t0 = time.time()
    results = [poll_until_terminal(base_url, jid) for jid in job_ids]
    wall_time = time.time() - t0

    processing_latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    dead = sum(1 for r in results if r["status"] == "dead_letter")

    print(f"\ndone in {wall_time:.1f}s")
    print(f"submit p50/p95/p99 (ms): "
          f"{statistics.median(submit_latencies):.1f} / "
          f"{_pctile(submit_latencies, 95):.1f} / "
          f"{_pctile(submit_latencies, 99):.1f}")
    if processing_latencies:
        print(f"inference p50/p95/p99 (ms): "
              f"{statistics.median(processing_latencies):.1f} / "
              f"{_pctile(processing_latencies, 95):.1f} / "
              f"{_pctile(processing_latencies, 99):.1f}")
    print(f"dead-lettered: {dead}/{count}")


def run_steady(base_url: str, jobs_per_day: int, image_path: str, duration_s: float) -> None:
    interval = 86400 / jobs_per_day
    print(f"steady-state: 1 job every {interval:.1f}s ({jobs_per_day}/day) for {duration_s:.0f}s")
    end = time.time() + duration_s
    n = 0
    while time.time() < end:
        job_id, lat = submit_one(base_url, image_path)
        n += 1
        print(f"  [{n}] submitted {job_id} (submit latency {lat:.1f}ms)")
        time.sleep(interval)
    print(f"submitted {n} jobs")


def _pctile(values: list[float], pct: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--image", required=True, help="path to a sample jpg/png to submit repeatedly")
    parser.add_argument("--mode", choices=["burst", "steady"], default="burst")
    parser.add_argument("--count", type=int, default=100, help="burst mode: total jobs to submit")
    parser.add_argument("--jobs-per-day", type=int, default=1500, help="steady mode: target daily rate")
    parser.add_argument("--duration", type=float, default=300, help="steady mode: how long to run, seconds")
    args = parser.parse_args()

    if args.mode == "burst":
        run_burst(args.base_url, args.count, args.image)
    else:
        run_steady(args.base_url, args.jobs_per_day, args.image, args.duration)
