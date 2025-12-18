# Multi-arch Dockerfile for the FastAPI app
# Builds on linux/amd64 and linux/arm64 using docker buildx

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bullseye

# Do not run as root
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by common data and ML libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
        curl \
        libatlas3-base \
        liblapack3 \
        libblas3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directory
RUN useradd --create-home app
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Use pip options to prefer binary wheels when available
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# Copy source
COPY . /app
RUN chown -R app:app /app
USER app

EXPOSE 8000

# Use Uvicorn to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
