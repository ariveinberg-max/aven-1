# Reproducible ML environment (CPU). The GPU variant (CUDA base image) is WP-3.3.
#   docker build -f docker/ml.Dockerfile -t neurolayer-ml .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" neurolayer-ml neurolayer smoke
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra neuro --extra tracking --no-install-project
COPY src ./src
COPY catalog ./catalog
COPY configs ./configs
RUN uv sync --locked --no-dev --extra neuro --extra tracking
RUN useradd --create-home --uid 10001 ml && chown -R ml /app
USER ml
ENV PATH="/app/.venv/bin:$PATH"
CMD ["neurolayer", "--help"]
