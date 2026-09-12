"""
Minimal storage abstraction. In-cluster, api and worker pods mount the same
ReadWriteMany PVC at IMAGE_STORAGE_DIR (see k8s/pvc.yaml), so the API writes
the uploaded image once and hands the worker a path instead of round-tripping
image bytes through Redis. Swap this for an S3/GCS client in production
without touching the queue logic.
"""
from __future__ import annotations

import os
import uuid

from fastapi import UploadFile

from tensorqueue_common import settings

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ext(filename: str) -> str:
    _, ext = os.path.splitext(filename or "")
    return ext.lower() if ext.lower() in ALLOWED_EXTENSIONS else ".jpg"


async def save_upload(file: UploadFile) -> str:
    os.makedirs(settings.image_storage_dir, exist_ok=True)
    dest_name = f"{uuid.uuid4().hex}{_ext(file.filename)}"
    dest_path = os.path.join(settings.image_storage_dir, dest_name)
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)
    return dest_path
