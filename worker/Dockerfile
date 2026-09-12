FROM python:3.11-slim

WORKDIR /srv

# Runtime libs required by opencv-python (an ultralytics dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY common /srv/common
RUN pip install --no-cache-dir /srv/common

COPY worker/requirements.txt /srv/worker/requirements.txt
# CPU-only torch build by default — swap the index url for a CUDA build
# when targeting GPU node pools.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /srv/worker/requirements.txt

COPY worker/*.py /srv/worker/
WORKDIR /srv/worker

ENV PYTHONUNBUFFERED=1
EXPOSE 9000

RUN useradd -m appuser && mkdir -p /data/images && chown -R appuser /data/images
USER appuser

CMD ["python", "worker.py"]
