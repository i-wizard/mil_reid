# syntax=docker/dockerfile:1
#
# Portable CPU image for the animal re-identification ML core.
#
# Pinned to linux/amd64 so the `+cpu` torch wheels in requirements.txt resolve
# deterministically regardless of the reviewer's host architecture. The image is
# self-contained (code is COPYed in), while docker-compose overlays the source
# dirs as bind mounts for live editing — see docker-compose.yml.
FROM --platform=linux/amd64 python:3.12-slim

# Fail fast and keep logs unbuffered so `docker compose up` streams progress; no
# pip cache layer since the image is built once and shipped.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Send the HuggingFace cache (MegaDescriptor weights) into the mounted
    # artifacts volume so the multi-hundred-MB download survives across runs.
    HF_HOME=/app/artifacts/hf_cache

# opencv-python (pulled in by wildlife-datasets) needs libGL + glib at runtime;
# the slim base omits them. Install only those, then drop apt lists to stay small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first, as their own layer, so code edits don't bust the
# (slow) dependency layer on rebuild.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the source in so the image runs standalone (compose will overlay these
# with bind mounts for live development).
COPY ml ./ml
COPY scripts ./scripts
COPY tests ./tests
COPY api ./api
COPY web ./web

# The API/UI listen here; compose publishes it. (Documentation only — EXPOSE does
# not publish by itself.)
EXPOSE 8000

# Default to the dataset-free smoke test: `docker compose up` proves the
# environment works without downloading anything. Override for pipeline steps,
# e.g. `docker compose run --rm reid python -m scripts.train`. The `api` service
# overrides this with the uvicorn command.
CMD ["python", "-m", "tests.test_core_logic"]
