# Building and Pushing Multi-arch Docker Images

This project includes a Dockerfile and a GitHub Actions workflow to build and push multi-arch images (linux/amd64 and linux/arm64) to Docker Hub.

## Required secrets (GitHub Actions)
- `DOCKERHUB_USERNAME` - your Docker Hub username
- `DOCKERHUB_TOKEN` - a Docker Hub access token (store in GitHub Secrets)
- `DOCKERHUB_REPO` - repository name (e.g. `csv_gen` or `username/csv_gen`)

## How the workflow works
- The workflow file is at `.github/workflows/docker-publish.yml` and uses `docker/build-push-action` with `platforms: linux/amd64,linux/arm64`.
- It builds using the `be/Dockerfile` context and pushes `username/${{ secrets.DOCKERHUB_REPO }}:latest` and a short SHA tag.

## Local multi-arch build & push (convenience)
A helper script is available at `be/build_and_push.sh` to build and push both the backend and frontend images.

There are convenience files for deploying and running on remote hosts:
- `docker_cloud/docker-compose.yml` — production compose file (reads `BACKEND_IMAGE` and `FRONTEND_IMAGE` from env)
- `docker_cloud/.env.example` — sample env file for production
- `scripts/pull_and_run.sh` — helper that pulls configured images and starts the compose stack

There is an example env file at `be/.env.example`. Copy it to `be/.env` and edit it to avoid passing `IMAGE` on the command line:

  cp be/.env.example be/.env

Example usage:

1. Login to Docker Hub:

   docker login -u $DOCKERHUB_USERNAME -p $DOCKERHUB_TOKEN

2. Build & push both images (backend + frontend) using a single base name:

   IMAGE=yourusername/yourrepo ./be/build_and_push.sh

   This will push:
   - backend -> yourusername/yourrepo:latest
   - frontend -> yourusername/yourrepo-frontend:latest

   Or set images explicitly:

   BACKEND_IMAGE=yourname/backend:tag FRONTEND_IMAGE=yourname/frontend:tag ./be/build_and_push.sh

3. On the remote host (example):

   # copy docker_cloud/.env.example -> docker_cloud/.env and set the two image variables
   docker compose -f docker_cloud/docker-compose.yml pull
   docker compose -f docker_cloud/docker-compose.yml up -d

Notes:
- The `--no-push` / `-n` flag will build images locally without pushing; useful for testing.
- Ensure `ALLOWED_HOSTS` is set in `docker_cloud/.env` or via environment when running in production to allow the frontend domain.

Notes:
- Script requires `docker buildx` support (modern Docker Desktop includes this).
- The script will create a `buildx` builder if needed and push the multi-arch image to Docker Hub.
- `be/build_and_push.sh` will load `be/.env` automatically if present (do not commit `be/.env` with secrets).

## Important: TensorFlow / arm64 compatibility
- The repo depends on `tensorflow` (via `requirements.txt`) for LSTM-based forecasting. Prebuilt TensorFlow wheels for linux/arm64 are not always available for all TensorFlow versions.

Options if arm64 build fails due to TensorFlow:
- Build and publish only `linux/amd64` (set `platforms: linux/amd64` in the workflow).
- Use an **inference-only** image that does not install the full training dependencies (remove `tensorflow` from the image and use a specialized inference runtime such as TensorFlow Serving or ONNX runtime). This is typically much smaller and more reliable on arm64.
- Locate or build an aarch64-compatible TensorFlow wheel (community provided `tensorflow-aarch64` builds exist for some versions) and add instructions or conditional install to the Dockerfile.

## Health checks and running
- The container exposes port `8000` and runs Uvicorn serving `main:app` (from the `be` folder).
- Example run:

  docker run -p 8000:8000 yourusername/yourrepo:latest

## Troubleshooting
- If a build fails during `pip install tensorflow`, try customizing the Dockerfile to pin a TF version known to have arm64 wheels, or build for `linux/amd64` only.
- For CI failures, check Actions logs for the pip error / missing wheel details.

If you want, I can:
- Add an inference-only Dockerfile variant (smaller image) and workflow for it, or
- Add conditional logic to the Dockerfile to support alternative TF wheel sources. Let me know which option you prefer.
