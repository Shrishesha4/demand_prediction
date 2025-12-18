#!/usr/bin/env bash
set -euo pipefail

# Build and push backend and frontend multi-arch images.
# Usage:
#   IMAGE=yourname/repo ./build_and_push.sh          # will push backend -> IMAGE:latest and frontend -> IMAGE-frontend:latest
#   BACKEND_IMAGE=yourname/backend FRONTEND_IMAGE=yourname/frontend ./build_and_push.sh
#   ./build_and_push.sh --no-push                    # build locally (no push), requires docker build

# Load .env from script directory if present (optional). DO NOT commit secrets.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
fi

# Parse flags
NO_PUSH=false
for arg in "${@-}"; do
  case "$arg" in
    --no-push|-n)
      NO_PUSH=true
      shift || true
      ;;
    --help|-h)
      echo "Usage: IMAGE=base/repo BACKEND_IMAGE=... FRONTEND_IMAGE=... ./build_and_push.sh [--no-push]"
      echo "If BACKEND_IMAGE/FRONTEND_IMAGE not set, defaults: BACKEND_IMAGE=\$IMAGE:latest FRONTEND_IMAGE=\$IMAGE-frontend:latest"
      exit 0
      ;;
  esac
done

# Derive image names
IMAGE_BASE=${IMAGE:-}
BACKEND_IMAGE=${BACKEND_IMAGE:-}
FRONTEND_IMAGE=${FRONTEND_IMAGE:-}

if [[ -z "$BACKEND_IMAGE" && -z "$IMAGE_BASE" ]]; then
  echo "Please set BACKEND_IMAGE or IMAGE (base) — e.g. IMAGE=username/repo"
  exit 1
fi

if [[ -z "$BACKEND_IMAGE" ]]; then
  BACKEND_IMAGE="${IMAGE_BASE}:latest"
fi
if [[ -z "$FRONTEND_IMAGE" ]]; then
  # default frontend image is base + '-frontend'
  if [[ -n "$IMAGE_BASE" ]]; then
    FRONTEND_IMAGE="${IMAGE_BASE}-frontend:latest"
  else
    FRONTEND_IMAGE="${BACKEND_IMAGE%:*}-frontend:${BACKEND_IMAGE#*:}"
  fi
fi

# ensure Docker daemon is available
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon not running. Please start Docker Desktop or run 'colima start --with-docker'."
  exit 1
fi

# ensure buildx is ready for multi-arch builds
if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx CLI plugin not found. Install it: https://github.com/docker/buildx/releases"
  exit 1
fi

docker buildx create --name multiarch-builder --use || true
docker buildx inspect --bootstrap >/dev/null 2>&1 || true

# Build helper
build_and_maybe_push() {
  local context="$1"; shift
  local dockerfile="$1"; shift
  local image="$1"; shift
  if [[ "$NO_PUSH" == "true" ]]; then
    echo "Building (local) ${image} from ${context} (${dockerfile})"
    docker build -t "${image}" -f "${dockerfile}" "${context}"
  else
    echo "Building & pushing ${image} (multi-arch) from ${context} (${dockerfile})"
    docker buildx build \
      --platform linux/amd64,linux/arm64 \
      -t "${image}" \
      -f "${dockerfile}" \
      --push \
      "${context}"
  fi
}

# Build backend
echo "-- Backend: ${BACKEND_IMAGE}"
build_and_maybe_push "${SCRIPT_DIR}" "${SCRIPT_DIR}/Dockerfile" "${BACKEND_IMAGE}"

# Build frontend (gui)
FRONTEND_DIR="$(cd "${SCRIPT_DIR}/.." && cd gui && pwd)"
if [[ ! -f "${FRONTEND_DIR}/Dockerfile.prod" ]]; then
  echo "Frontend Dockerfile not found at ${FRONTEND_DIR}/Dockerfile.prod"
  exit 1
fi

echo "-- Frontend: ${FRONTEND_IMAGE} (context: ${FRONTEND_DIR})"
build_and_maybe_push "${FRONTEND_DIR}" "${FRONTEND_DIR}/Dockerfile.prod" "${FRONTEND_IMAGE}"

# Provide next steps
if [[ "$NO_PUSH" == "true" ]]; then
  echo "Built images locally. You can run them with your docker-compose files or tag+push manually."
else
  echo "Pushed images:"
  echo "  Backend -> ${BACKEND_IMAGE}"
  echo "  Frontend -> ${FRONTEND_IMAGE}"
  echo "On remote hosts: set BACKEND_IMAGE and FRONTEND_IMAGE env vars (or update docker_cloud/docker-compose.yml defaults), then run:"
  echo "  docker compose -f docker_cloud/docker-compose.yml pull && docker compose -f docker_cloud/docker-compose.yml up -d"
fi

echo "Done"
