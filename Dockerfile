# syntax=docker/dockerfile:1.20

# BUILDER STAGE
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/root/.cache/uv \
    apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev

# RUNTIME STAGE
FROM python:3.12-slim-bookworm
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /app

COPY --from=builder --chown=nonroot:nonroot /app/.venv ./.venv
COPY --chown=nonroot:nonroot src ./src
COPY --chown=nonroot:nonroot experiments ./experiments

ENV PATH="/app/.venv/bin:$PATH"
USER nonroot
