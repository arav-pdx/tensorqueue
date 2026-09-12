"""
Centralized configuration, driven entirely by environment variables so the
same image can run in docker-compose, kind, or a real cluster without
rebuilding. Mirrors what would normally live in a K8s ConfigMap + Secret.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Redis / queue
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: str | None = os.getenv("REDIS_PASSWORD") or None

    # Queue keys
    queue_pending: str = os.getenv("QUEUE_PENDING_KEY", "tensorqueue:pending")
    queue_delayed: str = os.getenv("QUEUE_DELAYED_KEY", "tensorqueue:delayed")
    queue_dlq: str = os.getenv("QUEUE_DLQ_KEY", "tensorqueue:dlq")
    queue_processing: str = os.getenv("QUEUE_PROCESSING_KEY", "tensorqueue:processing")
    job_hash_prefix: str = os.getenv("JOB_HASH_PREFIX", "tensorqueue:job:")

    # Retry / backoff policy
    max_retries: int = int(os.getenv("MAX_RETRIES", "5"))
    base_backoff_seconds: float = float(os.getenv("BASE_BACKOFF_SECONDS", "2"))
    max_backoff_seconds: float = float(os.getenv("MAX_BACKOFF_SECONDS", "300"))
    backoff_jitter_seconds: float = float(os.getenv("BACKOFF_JITTER_SECONDS", "1"))

    # Job lifetime
    job_ttl_seconds: int = int(os.getenv("JOB_TTL_SECONDS", str(60 * 60 * 24)))  # 24h
    processing_timeout_seconds: int = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "60"))

    # Storage for uploaded images (shared volume / PVC path in-cluster)
    image_storage_dir: str = os.getenv("IMAGE_STORAGE_DIR", "/data/images")

    # Worker / model
    model_weights: str = os.getenv("MODEL_WEIGHTS", "yolov8n.pt")
    model_device: str = os.getenv("MODEL_DEVICE", "cpu")  # "cpu" | "cuda" | "cuda:0"
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    worker_poll_timeout_seconds: int = int(os.getenv("WORKER_POLL_TIMEOUT_SECONDS", "5"))
    requeue_scan_interval_seconds: float = float(os.getenv("REQUEUE_SCAN_INTERVAL_SECONDS", "1.0"))

    # Metrics
    metrics_port: int = int(os.getenv("METRICS_PORT", "9000"))
    latency_window_size: int = int(os.getenv("LATENCY_WINDOW_SIZE", "2000"))  # samples kept for p99


settings = Settings()
