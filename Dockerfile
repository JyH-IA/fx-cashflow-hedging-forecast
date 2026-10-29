# syntax=docker/dockerfile:1
# TODO(verify): pin the base images by digest (python:3.11-slim@sha256:...) after checking them.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/app
# libgomp1 is required by LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app app ./app
COPY --chown=app:app configs ./configs
COPY --chown=app:app sql ./sql
COPY --chown=app:app .streamlit ./.streamlit
# Models and data are mounted at runtime (-v ./models:/app/models:ro -v ./data:/app/data:ro).
# Secrets are passed at runtime (--env-file .env); never with ENV or ARG.
USER app
EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1
# For the API instead: uvicorn fxhedge.api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
