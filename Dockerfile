# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/ ./agents/

# Default command – run the agent with a sample query.
# Override CMD or set the QUERY environment variable as needed.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "agents.agent"]
CMD []
