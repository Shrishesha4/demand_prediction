# E-commerce Demand Forecasting

A small project that compares classical time series (SARIMAX/ARIMA) with Deep Learning (LSTM) for e-commerce demand forecasting. It includes a FastAPI backend and a Svelte frontend (Vite) with Docker support for development and production deployment.

---

## 🚀 Quick links
- Backend: `be/` (FastAPI app)
- Frontend: `gui/` (Svelte + Vite)
- Dev Docker Compose: `docker-compose.yml` (local/dev)
- Cloud/Prod Compose: `docker_cloud/docker-compose.yml`
- Build & push helper: `be/build_and_push.sh`
- Remote pull-and-run helper: `scripts/pull_and_run.sh`

---

## Features
- Train and compare SARIMAX and LSTM models
- LSTM model training, persistence and prediction endpoints
- Multi-arch Docker builds (amd64 + arm64) via `docker buildx`
- Frontend UI for uploading datasets, visualizing results and downloading trained model
- Optional production-oriented compose with environment-driven configuration

---

## Prerequisites
- Docker (Desktop or daemon) and `docker buildx` available
- Node.js (>= 20 recommended for building the frontend)
- Python 3.11 (project uses this in the Dockerfile)
- (Optional) `colima` if you prefer not to run Docker Desktop on macOS

---

## Local development

### Backend (local, venv)

1. Create and activate a Python virtual env:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies and run the API:

```bash
pip install -r be/requirements.txt
cd be
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Health and docs:
- Health: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

### Frontend (dev)

1. Install dependencies and run dev server:

```bash
cd gui
npm i
npm run dev
```

2. Open the UI at `http://localhost:5173` (or the printed host/port). The dev server will accept configured hostnames via `ALLOWED_HOSTS` (see the Vite section below).

---

## Docker (local dev)

Start both services with the local dev compose:

```bash
docker compose up -d --build
```

- Backend: `http://localhost:8000`
- Frontend (prod-like preview): `http://localhost:3030` (mapped to the preview/served port)

Logs and health:
```bash
docker compose logs -f demand-api
docker compose logs -f frontend
docker compose ps
```

---

## Building & Pushing Images (CI or locally)

Use the helper script `be/build_and_push.sh` which supports multi-arch `docker buildx` builds for both **backend** and **frontend** images.

Examples:

- Build & push using a base image name (defaults):

```bash
IMAGE=yourusername/yourrepo ./be/build_and_push.sh
```

This will push:
- backend -> `yourusername/yourrepo:latest`
- frontend -> `yourusername/yourrepo-frontend:latest`

- Explicit image names:

```bash
BACKEND_IMAGE=yourname/backend:tag FRONTEND_IMAGE=yourname/frontend:tag ./be/build_and_push.sh
```

- Build locally (no push):

```bash
IMAGE=yourname/yourrepo ./be/build_and_push.sh --no-push
```

> Notes: `docker buildx` must be available and the Docker daemon must be running.

---

## Production: pull & run on another host

1. Copy `docker_cloud/.env.example` -> `docker_cloud/.env` and set `BACKEND_IMAGE` and `FRONTEND_IMAGE` to the images you pushed.

2. Pull images & start stack (helper):

```bash
# from repo root on the remote host
./scripts/pull_and_run.sh
```

or manually:

```bash
BACKEND_IMAGE=... FRONTEND_IMAGE=... docker compose -f docker_cloud/docker-compose.yml pull
docker compose -f docker_cloud/docker-compose.yml up -d
```

Your frontend will bind port 80 (or the port configured in the compose file) and proxy API calls to the backend service.

---

## Vite / Frontend host configuration

- The preview server reads `ALLOWED_HOSTS` and `HOST` from environment variables. Set these in `docker_cloud/.env` or via environment when running the container.
- Example: `ALLOWED_HOSTS=pdp.s4home.dpdns.org,localhost` and `HOST=0.0.0.0`.
- For development only you can set `ALLOWED_HOSTS=all` but avoid exposing an unrestricted preview server on public networks.

Files: `gui/vite.config.ts` - parses `ALLOWED_HOSTS` and applies them to `server` and `preview` configurations.

---

## Troubleshooting

Common issues and fixes:

- "docker: unknown command: docker buildx" → install the Docker CLI Buildx plugin or update Docker Desktop.
- "failed to connect to the docker API" → start Docker Desktop or `colima start --with-docker`.
- Frontend blocked host message: add your domain to `ALLOWED_HOSTS` or run with `ALLOWED_HOSTS=all` in dev (see Vite section).
- Frontend build fails due to Node version: ensure Node >= 20 for Vite/SvelteKit builds (Dockerfile uses Node 20).
- TensorFlow on arm64: prebuilt wheels may not exist for some TF versions; consider building for `linux/amd64` only, or use an inference-only image without `tensorflow`.

---

## Project structure

```
.
├─ be/                     # FastAPI backend (Python)
│  ├─ main.py
│  ├─ forecast_pipeline.py
│  ├─ requirements.txt
│  └─ build_and_push.sh
├─ gui/                    # Svelte frontend (Vite)
│  ├─ src/
│  ├─ Dockerfile.prod
│  └─ vite.config.ts
├─ docker-compose.yml      # Local development compose
├─ docker_cloud/           # Production compose + .env.example
├─ scripts/pull_and_run.sh # Pull & run helper for remote hosts
└─ README.md
```
---