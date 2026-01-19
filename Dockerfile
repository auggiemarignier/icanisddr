# syntax=docker/dockerfile:1.20

# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Install the project into `/app`
WORKDIR /app

COPY --parents=true --chown=nonroot:nonroot pyproject.toml uv.lock src ./

RUN uv sync --frozen

CMD ["uv", "run", "pytest"]
