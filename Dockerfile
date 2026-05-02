# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/ ./agents/
COPY api/ ./api/
COPY frontend/ ./frontend/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default: run the web server
# Override CMD to run the CLI agent instead, e.g.:
#   docker run stonks.ai python -m agents.agent "What is AAPL?"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
