FROM python:3.11-slim

WORKDIR /srv

# Install the shared queue/config/models package first (better layer caching)
COPY common /srv/common
RUN pip install --no-cache-dir /srv/common

COPY api/requirements.txt /srv/api/requirements.txt
RUN pip install --no-cache-dir -r /srv/api/requirements.txt

COPY api/app /srv/app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Not root — matches the securityContext.runAsNonRoot in the k8s Deployment
RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
