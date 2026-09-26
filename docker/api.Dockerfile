# API image: core + api + dl extras (PyTorch is needed to serve the model). Build from the repo root:
#   docker build -f docker/api.Dockerfile -t neurolayer-api .
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first (cached layer), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra api --extra dl --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra api --extra dl

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 app
USER app
# Model bundles are mounted read-only at runtime (never baked into the image).
ENV NEUROLAYER_MODELS_DIR=/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["/app/.venv/bin/uvicorn", "neurolayer_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
