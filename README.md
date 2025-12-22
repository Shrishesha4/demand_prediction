# E-commerce Demand Forecasting

A small project that compares classical time series (SARIMAX/ARIMA) with Deep Learning (LSTM) for e-commerce demand forecasting. It includes a FastAPI backend and a Svelte frontend (Vite) with Docker support for development and production deployment.

---

## Quick links
- Backend: `be/` (FastAPI app)
- Frontend: `gui/` (Svelte + Vite)
- Dev Docker Compose: `docker-compose.yml` (local/dev)
- Cloud/Prod Compose: `docker_cloud/docker-compose.yml`
- Build & push helper: `build_and_push.sh` (root-level). The old `be/build_and_push.sh` is now a shim that forwards to the root script.
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

1. Install dependencies and run dev server (the GUI will respect the project root `.env` automatically):

```bash
cd gui
npm i
npm run dev
```

> The `gui` dev scripts now load environment variables from the project root `.env` (via `dotenv-cli`) so `HMR_PROTOCOL`, `FRONTEND_URL`, `HMR_CLIENT_PORT`, etc. are applied when present.

2. Open the UI at `http://localhost:5173` (or the printed host/port). The dev server will accept configured hostnames via `ALLOWED_HOSTS` (see the Vite section below).

---

## Docker (local dev)

Start both services with the local dev compose:

```bash
docker compose up -d --build
```

- Backend: `http://localhost:8000`
- Frontend (prod-like preview): `http://localhost:5173` (mapped to the preview/served port)

Logs and health:
```bash
docker compose logs -f demand-api
docker compose logs -f frontend
docker compose ps
```

---

## Building & Pushing Images (CI or locally)

Use the root helper script `build_and_push.sh` (now located at the project root). It builds **both backend and frontend** images and supports multi-arch `docker buildx` builds; it will fall back to single-arch builds if `buildx` is not available.

Examples:

- Build & push using a base image name (defaults):

```bash
IMAGE=yourusername/yourrepo ./build_and_push.sh
```

This will push:
- backend -> `yourusername/yourrepo:latest`
- frontend -> `yourusername/yourrepo-frontend:latest`

- Explicit image names:

```bash
BACKEND_IMAGE=yourname/backend:tag FRONTEND_IMAGE=yourname/frontend:tag ./build_and_push.sh
```

- Build locally (no push):

```bash
./build_and_push.sh --no-push
```

- Build only backend or frontend:

```bash
# backend only
./build_and_push.sh --backend-only
# frontend only
./build_and_push.sh --frontend-only
```

Notes:
- The script detects whether `docker buildx` is available and uses it for multi-arch builds; if not present it will perform single-arch builds and push the result (unless `--no-push` is used).
- The older `be/build_and_push.sh` remains for backwards compatibility and forwards to the root script.
- Docker daemon must be running (Docker Desktop, Colima, etc.).

---

## Production: pull & run on another host

1. Copy `docker_cloud/.env.example` -> `.env` at the project root and set `BACKEND_IMAGE`, `FRONTEND_IMAGE` and optionally `BACKEND_PORT`/`FRONTEND_PORT` to the images and ports you want to expose.

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

Notes:
- The `docker_cloud/docker-compose.yml` file now uses parameterized ports and environment variables (e.g. `${BACKEND_PORT:-8000}` and `${FRONTEND_PORT:-3030}`) and no longer hardcodes `container_name`, making it safer to run on shared hosts.
- The frontend service (image) is expected to serve the built UI on container port `80` by default; the compose mapping exposes it as `FRONTEND_PORT` on the host.
- If you use a reverse proxy (nginx, Traefik, Nginx Proxy Manager) in front of the stack, configure it to forward `/api/` to the backend and to proxy websocket upgrades for the dev HMR server if you are running in dev mode.
- See the Vite / HMR and Nginx sections below for exact proxy config snippets to support HMR and SPA reloads.

---

## Vite / Frontend host configuration & HMR (dev)

- The preview server reads `ALLOWED_HOSTS` and `HOST` from environment variables. Set these in the project root `.env` or via environment when running the container.
- Example: `ALLOWED_HOSTS=pdp.s4home.dpdns.org,localhost` and `HOST=0.0.0.0`.
- For development only you can set `ALLOWED_HOSTS=all` but avoid exposing an unrestricted preview server on public networks.

HMR and Proxy notes (dev behind a reverse proxy):
- Vite has separate concepts for the HMR server bind address (where it listens) and the client-facing address/port the browser connects to.
- We added environment-driven settings in `gui/vite.config.ts`:
  - `PUBLIC_HOST` (the public hostname the browser uses, e.g. `pdp.shrishesha.space`)
  - `HMR_PROTOCOL` (`wss` when using TLS) and `HMR_BIND_HOST` (server bind host, e.g. `0.0.0.0`)
  - `HMR_PORT` (server bind port, usually `5173`) and `HMR_CLIENT_PORT` (port clients connect to, usually `443` when proxied via TLS)

Run example (dev, proxied via Nginx Proxy Manager):
```bash
PUBLIC_HOST=pdp.shrishesha.space HMR_PROTOCOL=wss HMR_BIND_HOST=0.0.0.0 HMR_PORT=5173 HMR_CLIENT_PORT=443 HOST=0.0.0.0 npm run dev
```

Proxy configuration requirements (Nginx or Nginx Proxy Manager):
- Ensure the proxy forwards WebSocket upgrades and uses `proxy_http_version 1.1`.
- Replace any `proxy_set_header Connection $http_connection;` with `proxy_set_header Connection 'upgrade';` to guarantee the upgrade handshake reaches Vite.
- If Cloudflare sits in front of your proxy, set SSL mode to `Full` or `Full (strict)` and ensure the origin certificate is configured correctly.

Files: `gui/vite.config.ts` reads the above env vars and configures HMR; the frontend now uses a relative `/api/` prefix for API calls (see API section below).

---

## Troubleshooting & common fixes

- "docker: unknown command: docker buildx" → install the Docker CLI Buildx plugin or allow the script to fall back to single-arch builds (the helper now automatically falls back and will push the single-arch image if `buildx` is not available).
- "failed to connect to the docker API" → start Docker Desktop or `colima start --with-docker`.
- HMR WebSocket fails (e.g. `WebSocket closed without opened`):
  1. Ensure Vite is started with `HMR_BIND_HOST=0.0.0.0` and `HMR_CLIENT_PORT=443` (when using wss behind TLS). Example:
     ```bash
     PUBLIC_HOST=pdp.shrishesha.space HMR_PROTOCOL=wss HMR_BIND_HOST=0.0.0.0 HMR_PORT=5173 HMR_CLIENT_PORT=443 HOST=0.0.0.0 npm run dev
     ```
  2. Ensure your proxy (Nginx / Nginx Proxy Manager) has Websockets support enabled and includes:
     ```nginx
     proxy_http_version 1.1;
     proxy_set_header Upgrade $http_upgrade;
     proxy_set_header Connection 'upgrade';
     proxy_buffering off;
     ```
  3. If DNS is proxied through Cloudflare, temporarily switch the record to DNS-only (grey cloud) to confirm Cloudflare isn't causing the problem; if DNS-only works, set Cloudflare SSL to `Full` or `Full (strict)` and install an origin cert.
- SPA reloads returning 405 on non-root routes (e.g. `GET /predict` → `405`):
  - The reverse proxy must return the SPA `index.html` for GET requests on client routes and only proxy API/non-GET requests to the backend. See `gui/nginx.conf` for the implemented approach.
- Frontend blocked host message: add your domain to `ALLOWED_HOSTS` or run with `ALLOWED_HOSTS=all` in dev (see Vite section).
- Frontend build fails due to Node version: ensure Node >= 20 for Vite/SvelteKit builds (Dockerfile uses Node 20).
- TensorFlow on arm64: prebuilt wheels may not exist for some TF versions; consider building for `linux/amd64` only, or use an inference-only image without `tensorflow`.
- If you see `ERR_BLOCKED_BY_CLIENT` in the browser, try an incognito window or disable browser extensions (adblockers/privacy extensions often cause this).

If you need a hand, paste terminal logs (Vite server output, NPM/nginx error logs, and browser console network/WS traces) and I can help diagnose further.

---

## Project structure

```
.
├─ be/                     # FastAPI backend (Python)
│  ├─ main.py
│  ├─ forecast_pipeline.py
│  ├─ requirements.txt
│  └─ build_and_push.sh    # shim that forwards to root `build_and_push.sh`
├─ gui/                    # Svelte frontend (Vite)
│  ├─ src/
│  ├─ Dockerfile.prod
│  └─ vite.config.ts
├─ docker-compose.yml      # Local development compose (builds from local source)
├─ docker_cloud/           # Production compose + .env.example (pull & run images)
├─ scripts/pull_and_run.sh # Pull & run helper for remote hosts
├─ build_and_push.sh       # root-level build & push helper (builds backend+frontend)
└─ README.md
```
---