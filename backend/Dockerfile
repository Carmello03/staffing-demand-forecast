# Build stage
FROM python:3.11-slim AS builder

WORKDIR /install
COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt

# Runtime stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ARTIFACT_DIR=/app/artifacts \
    MODEL_ARTIFACT_PREFIX=lightgbm

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY app/ .

# Copy model artifacts (built during evaluation).
# The app resolves these via MODEL_ARTIFACT_DIR env var at runtime;
# override that env var in Cloud Run if you store artifacts in GCS instead.
COPY artifacts/ ./artifacts/

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
