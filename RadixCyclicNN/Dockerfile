# RadixCyclicNN image: build the React frontend with Node, then ship a slim
# Python runtime that serves the JSON API together with the built frontend.
#
#   docker build -t radixcyclicnn .                          pure Python, zero dependencies
#   docker build --build-arg WITH_TORCH=1 -t radixcyclicnn . + torch (GPU / vectorised backend, adds a few GB)
#
# Behind a registry mirror, point the base images elsewhere:
#   --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.11-slim --build-arg NODE_IMAGE=mirror.gcr.io/library/node:22-alpine
#
# The container runs the `radixnet` CLI; the default command serves the API on
# port 8000 with the model and checkpoints under /data (a volume).
# See docker-compose.yml for the full stack and docker/entrypoint.sh for the
# optional checkpoint resume on start-up.

ARG PYTHON_IMAGE=python:3.11-slim
ARG NODE_IMAGE=node:22-alpine

# ---- Stage 1: frontend ------------------------------------------------------
FROM ${NODE_IMAGE} AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: runtime -------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
ARG WITH_TORCH=0
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RADIXNET_MODEL=/data/model.json \
    RADIXNET_CHECKPOINT_DIR=/data/checkpoints
WORKDIR /app

COPY pyproject.toml README.md DESIGN.md ./
COPY radixnet/ ./radixnet/
COPY tests/ ./tests/
COPY data/ ./data/
COPY docker/entrypoint.sh /usr/local/bin/radixnet-entrypoint
COPY --from=frontend /build/dist ./frontend/dist

RUN pip install . \
 && if [ "$WITH_TORCH" = "1" ]; then pip install torch; fi \
 && chmod +x /usr/local/bin/radixnet-entrypoint \
 && useradd --create-home --uid 1000 radixnet \
 && mkdir -p /data/checkpoints \
 && chown -R radixnet:radixnet /data

USER radixnet
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

ENTRYPOINT ["radixnet-entrypoint"]
CMD ["--model", "/data/model.json", "serve", "--host", "0.0.0.0", "--port", "8000", \
     "--frontend-dir", "/app/frontend/dist", "--checkpoint-dir", "/data/checkpoints"]
