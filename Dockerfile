# Deployable image for the prediction API and the Streamlit dashboard.
# Training (which needs the optional torch extra) is expected to run on the host
# or a separate job; the trained model bundle is mounted in at /app/models.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARF_MODEL_DIR=/app/models

# libgomp1 is required by LightGBM / XGBoost; curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

# Application code + config.
COPY configs ./configs
COPY service ./service
COPY dashboard ./dashboard

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default: the FastAPI prediction service. Override `command` for the dashboard.
CMD ["uvicorn", "air_raid_forecasting.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
