ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

RUN useradd --create-home app
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

COPY . /app
RUN chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
