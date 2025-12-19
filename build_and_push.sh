#!/usr/bin/env bash
set -euo pipefail

# Build and push backend and frontend images from project root.
# Usage:
#   ./build_and_push.sh [--no-push] [--backend-only] [--frontend-only] [--help]
#   Or set env vars to override image names: BACKEND_IMAGE, FRONTEND_IMAGE
#   Optionally set IMAGE to a base (e.g. IMAGE=username/repo) to derive defaults.
# Examples:
#   ./build_and_push.sh --no-push
#   BACKEND_IMAGE=ghcr.io/me/api FRONTEND_IMAGE=ghcr.io/me/gui ./build_and_push.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow an optional .env in repo root
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
fi

# Defaults / flags
NO_PUSH=false
BUILD_BACKEND=true
BUILD_FRONTEND=true
for arg in "${@-}"; do
  case "$arg" in
    --no-push|-n)
      NO_PUSH=true
      shift || true
      ;;
    --backend-only)
      BUILD_FRONTEND=false
      shift || true
      ;;
    --frontend-only)
      BUILD_BACKEND=false
      shift || true
      ;;
    --help|-h)
      sed -n '1,120p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
  esac
done

# Derive image names (sensible defaults so script runs without env vars)
IMAGE_BASE=${IMAGE:-}
BACKEND_IMAGE=${BACKEND_IMAGE:-${IMAGE_BASE:-demand-api}:latest}
FRONTEND_IMAGE=${FRONTEND_IMAGE:-${IMAGE_BASE:-demand-frontend}-frontend:latest}

# Infer directories
BE_DIR="${SCRIPT_DIR}/be"
FRONTEND_DIR="${SCRIPT_DIR}/gui"

# If user provided only BACKEND_IMAGE but not FRONTEND_IMAGE above, ensure frontend
# uses a related tag when IMAGE_BASE isn't set.
if [[ -n "${BACKEND_IMAGE:-}" && -z "${FRONTEND_IMAGE:-}" ]]; then
  FRONTEND_IMAGE="${BACKEND_IMAGE%:*}-frontend:${BACKEND_IMAGE#*:}"
fi

# If we're going to push, require that image names contain a namespace (username/)
# to avoid accidentally attempting to push to the `library/` namespace on Docker Hub
# which typically requires special permissions. You can still build locally with
# --no-push which allows unnamespaced images.
if [[ "$NO_PUSH" != "true" ]]; then
  for var in BACKEND_IMAGE FRONTEND_IMAGE; do
    img="${!var}"
    if [[ "$img" != */* ]]; then
      echo "Error: image '$img' does not include a registry/namespace (e.g. 'username/repo:tag')."
      echo "Set $var to a valid remote image (e.g. yourhubuser/repo:tag) or run with --no-push." >&2
      exit 1
    fi
  done
fi

# ensure Docker daemon is available
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon not running. Please start Docker Desktop or run 'colima start --with-docker'."
  exit 1
fi

# detect whether buildx is available; if not, fall back to regular docker build
USE_BUILDX=true
if ! docker buildx version >/dev/null 2>&1; then
  echo "Warning: docker buildx not available — falling back to single-arch docker build/push"
  USE_BUILDX=false
else
  # Create or use an existing buildx builder in an idempotent way to avoid noisy errors
  if ! docker buildx inspect multiarch-builder >/dev/null 2>&1; then
    docker buildx create --name multiarch-builder --use || true
  else
    docker buildx use multiarch-builder || true
  fi
  docker buildx inspect --bootstrap >/dev/null 2>&1 || true
fi

build_and_maybe_push() {
  local context="$1"; shift
  local dockerfile="$1"; shift
  local image="$1"; shift
  if [[ "$NO_PUSH" == "true" ]]; then
    echo "Building (local) ${image} from ${context} (${dockerfile})"
    docker build -t "${image}" -f "${dockerfile}" "${context}"
  else
    if [[ "$USE_BUILDX" == "true" ]]; then
      echo "Building & pushing ${image} (multi-arch) from ${context} (${dockerfile})"
      docker buildx build \
        --platform linux/amd64,linux/arm64 \
        -t "${image}" \
        -f "${dockerfile}" \
        --push \
        "${context}"
    else
      echo "Building ${image} (single-arch) from ${context} (${dockerfile})"
      docker build -t "${image}" -f "${dockerfile}" "${context}"
      echo "Pushing ${image} to registry"
      docker push "${image}"
    fi
  fi
}

# Build backend
if [[ "$BUILD_BACKEND" == "true" ]]; then
  if [[ ! -f "${BE_DIR}/Dockerfile" ]]; then
    echo "Backend Dockerfile not found at ${BE_DIR}/Dockerfile"
    exit 1
  fi
  echo "-- Backend: ${BACKEND_IMAGE}"
  build_and_maybe_push "${BE_DIR}" "${BE_DIR}/Dockerfile" "${BACKEND_IMAGE}"
fi

# Build frontend
if [[ "$BUILD_FRONTEND" == "true" ]]; then
  if [[ ! -f "${FRONTEND_DIR}/Dockerfile.prod" ]]; then
    echo "Frontend Dockerfile not found at ${FRONTEND_DIR}/Dockerfile.prod"
    exit 1
  fi
  echo "-- Frontend: ${FRONTEND_IMAGE} (context: ${FRONTEND_DIR})"
  build_and_maybe_push "${FRONTEND_DIR}" "${FRONTEND_DIR}/Dockerfile.prod" "${FRONTEND_IMAGE}"
fi

# Provide next steps
if [[ "$NO_PUSH" == "true" ]]; then
  echo "Built images locally. You can run them with your docker-compose files or tag+push manually."
else
  echo "Pushed images:"
  [[ "$BUILD_BACKEND" == "true" ]] && echo "  Backend -> ${BACKEND_IMAGE}"
  [[ "$BUILD_FRONTEND" == "true" ]] && echo "  Frontend -> ${FRONTEND_IMAGE}"
  echo "On remote hosts: set BACKEND_IMAGE and FRONTEND_IMAGE env vars (or update docker_cloud/docker-compose.yml defaults), then run:" 
  echo "  docker compose -f docker_cloud/docker-compose.yml pull && docker compose -f docker_cloud/docker-compose.yml up -d"
fi

echo "Done"
