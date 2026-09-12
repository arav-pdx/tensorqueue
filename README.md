# TensorQueue — AI Inference Gateway & Job Queue Manager

A distributed, Kubernetes-native system that decouples a client-facing API
from an autoscaled fleet of YOLO inference workers via a fault-tolerant
Redis job queue.

```
client -> [FastAPI gateway] -> Redis (pending / delayed / processing / DLQ)
                                    ^                    |
                                    |                    v
                            [maintenance thread]   [YOLO worker pods]
                                                          |
                                                          v
                                                   shared image volume
```

## Why it's built this way

**Gateway/worker decoupling.** `api/` never imports torch and has no GPU
dependency — it's a thin, stateless I/O layer that writes an uploaded image
to shared storage, pushes a job id onto Redis, and returns `202 Accepted`
immediately. `worker/` never opens an HTTP port for client traffic. Each
tier scales, deploys, and fails independently: a bad model rollout can crash
every worker pod without taking the API down, and a traffic spike scales
the API pods without spinning up GPU capacity it doesn't need.

**Fault-tolerant queue (`common/tensorqueue_common/queue.py`).** Built on
plain Redis primitives (list + two sorted sets), so it needs nothing beyond
a stock Redis deployment:
- `pending` (LIST) — jobs ready to run, consumed via blocking `BRPOP`.
- `delayed` (ZSET, score = ready-at time) — retry backoff scheduling.
- `processing` (ZSET, score = visibility deadline) — in-flight jobs. If a
  worker pod is OOMKilled or evicted mid-inference, the job doesn't just
  disappear: a background maintenance loop reclaims anything past its
  deadline and routes it back through the normal retry path.
- `dlq` (LIST) — jobs that exhausted `MAX_RETRIES`, for inspection
  (`GET /v1/dlq`) and manual replay (`POST /v1/dlq/{id}/requeue`).

Retries use **exponential backoff with jitter**: `min(base * 2^retries,
max_backoff) + random_jitter`, so a downstream hiccup doesn't turn into a
thundering-herd retry storm against Redis or the model.

**Sub-100ms p99 latency.** `worker/inference.py` loads the YOLO model once
per pod and runs a warm-up inference at startup so the model, CUDA context,
and any JIT/kernel-autotuning overhead are paid before the first real job —
otherwise the *first* request after a pod starts would blow out p99, not
just p50. Every job's latency is recorded both into a capped Redis list
(cheap, cluster-wide p50/p95/p99 via `GET /v1/metrics`, no extra
infrastructure) and as a Prometheus histogram on each worker's `:9000/metrics`
for real dashboards/alerting.

**1.5K+ images/day at scale.** The worker `Deployment` in
`k8s/30-worker.yaml` is scaled by a KEDA `ScaledObject` on **queue depth**
(`tensorqueue:pending` list length), not CPU — a burst of uploads scales
worker replicas out before CPU utilization even reflects the backlog. A
CPU-based `HorizontalPodAutoscaler` fallback is included
(`k8s/30b-worker-hpa-cpu-fallback.yaml`) for clusters without KEDA.

## Repo layout

```
common/tensorqueue_common/   shared config, models, queue, metrics
                              (pip-installed into both images)
api/                          FastAPI gateway (no torch dependency)
worker/                       queue consumer + YOLO inference
k8s/                          namespace, config, redis, api, worker, ingress
tests/                        queue fault-tolerance tests (fakeredis, no infra)
scripts/load_test.py          burst + steady-state load generator
```

## Running it locally (no Kubernetes needed)

Requires Docker and a local Redis, or run everything with plain `docker run`:

```bash
# 1. Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 2. Shared image volume
mkdir -p /tmp/tq-images

# 3. Build images
docker build -t tensorqueue-api -f api/Dockerfile .
docker build -t tensorqueue-worker -f worker/Dockerfile .

# 4. Run the API
docker run -d --name tq-api -p 8000:8000 \
  -e REDIS_HOST=host.docker.internal -e IMAGE_STORAGE_DIR=/data/images \
  -v /tmp/tq-images:/data/images tensorqueue-api

# 5. Run a worker
docker run -d --name tq-worker -p 9000:9000 \
  -e REDIS_HOST=host.docker.internal -e IMAGE_STORAGE_DIR=/data/images \
  -v /tmp/tq-images:/data/images tensorqueue-worker

# 6. Submit a job
curl -F "file=@sample.jpg" http://localhost:8000/v1/jobs
curl http://localhost:8000/v1/jobs/<job_id>
curl http://localhost:8000/v1/metrics
```

Note: `worker/requirements.txt` pulls CPU PyTorch + Ultralytics, which is a
multi-hundred-MB image build — expect the first `docker build` for the
worker to take a while.

## Deploying to Kubernetes

```bash
# Build & push images to your registry, then update the `image:` fields in
# k8s/20-api.yaml and k8s/30-worker.yaml.

kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-config.yaml        # edit the Secret first!
kubectl apply -f k8s/02-storage.yaml       # requires an RWX-capable StorageClass
kubectl apply -f k8s/10-redis.yaml
kubectl apply -f k8s/20-api.yaml
kubectl apply -f k8s/30-worker.yaml        # requires KEDA — see the file header
kubectl apply -f k8s/40-ingress.yaml
```

No KEDA installed? Apply `k8s/30b-worker-hpa-cpu-fallback.yaml` instead of
the `ScaledObject` in `30-worker.yaml`.

For local cluster testing, `kind` or `minikube` both work — just point
`image:` at locally-built images (`kind load docker-image ...` /
`minikube image load ...`) and use `local-path`/`hostpath` storage classes
in place of RWX-capable ones for a single-node demo (multi-writer semantics
won't matter with one node).

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The test suite (`tests/test_queue.py`) exercises the fault-tolerance logic
directly against `fakeredis` — no real Redis or model required — covering:
enqueue/dequeue, successful completion, exponential backoff scheduling,
backoff growth across repeated failures, DLQ routing after exhausted
retries, manual DLQ requeue, and crash recovery (stale `processing` entries
being reclaimed).

## Load testing

```bash
pip install requests
python scripts/load_test.py --mode burst --count 200 --image sample.jpg
python scripts/load_test.py --mode steady --jobs-per-day 1500 --image sample.jpg
```

`burst` mode submits N jobs as fast as possible, polls every one to a
terminal state, and reports client-observed submit latency plus
worker-reported inference p50/p95/p99 — useful for confirming the
queue-depth-based autoscaling actually kicks in under load.

## Configuration reference

All config lives in `common/tensorqueue_common/config.py`, driven entirely
by environment variables (see `k8s/01-config.yaml` for the full list):
`MAX_RETRIES`, `BASE_BACKOFF_SECONDS`, `MAX_BACKOFF_SECONDS`,
`PROCESSING_TIMEOUT_SECONDS` (visibility timeout for crash detection),
`MODEL_DEVICE` (`cpu`/`cuda`), `CONFIDENCE_THRESHOLD`, and more.

## Known simplifications / what a production hardening pass would add

- Single-instance Redis in `k8s/10-redis.yaml` — swap for managed Redis
  (ElastiCache/MemoryStore) or Sentinel for HA; the app only depends on
  `REDIS_HOST`/`REDIS_PORT`, so this is a config change, not a code change.
- Shared-volume image storage assumes an RWX StorageClass; an
  S3/GCS-backed `storage.py` implementation would drop that requirement
  entirely and is a natural next step.
- No auth/rate-limiting on the API — add an API-key or OAuth2 dependency
  and a rate limiter (e.g. `slowapi`) before exposing this publicly.
- `MODEL_DEVICE=cuda` requires a GPU node pool with the NVIDIA device
  plugin installed and `nvidia.com/gpu` resource limits uncommented in
  `k8s/30-worker.yaml`.
