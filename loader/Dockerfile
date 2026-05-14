# Image for the `loader` service in docker-compose.yml.
# Runs push_data_into_minio.py + import_data.py inside the compose network,
# so it can talk to neo4j:7687 and minio:9000 directly while still storing
# host-reachable image_link URLs (http://localhost:9000/...).
#
# Note: PyTorch on CPU inside a container is noticeably slower than running
# on the host with MPS/CUDA. For one-shot ingestion this is acceptable.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

# System deps:
#   - curl for service-readiness checks during entrypoint
#   - libgl1 + libglib2.0-0 satisfy opencv-style transitive deps that
#     open_clip / pillow occasionally pull in via wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps from the shared requirements.txt. Cached as a single
# layer so a code-only change doesn't reinstall torch.
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY push_data_into_minio.py import_data.py /app/
COPY loader/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
