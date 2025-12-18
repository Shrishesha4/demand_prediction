#!/usr/bin/env bash
set -euo pipefail

# Pull images from registry and run docker compose for production cloud stack.
# Usage:
#   ./scripts/pull_and_run.sh           # will use BACKEND_IMAGE/FRONTEND_IMAGE from env or .env
#   BACKEND_IMAGE=org/backend:tag FRONTEND_IMAGE=org/frontend:tag ./scripts/pull_and_run.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker_cloud/docker-compose.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE"
  exit 1
fi

# Load .env if present in docker_cloud
if [[ -f "${ROOT_DIR}/docker_cloud/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/docker_cloud/.env"
fi

: ${BACKEND_IMAGE:="${BACKEND_IMAGE:-}"}
: ${FRONTEND_IMAGE:="${FRONTEND_IMAGE:-}"}

if [[ -z "$BACKEND_IMAGE" || -z "$FRONTEND_IMAGE" ]]; then
  echo "Please set BACKEND_IMAGE and FRONTEND_IMAGE environment variables or add them to docker_cloud/.env"
  echo "Example: BACKEND_IMAGE=shrishesha4/df:latest FRONTEND_IMAGE=shrishesha4/df-frontend:latest ./scripts/pull_and_run.sh"
  exit 1
fi

export BACKEND_IMAGE
export FRONTEND_IMAGE

echo "Pulling images: $BACKEND_IMAGE and $FRONTEND_IMAGE"
docker compose -f "$COMPOSE_FILE" pull

echo "Starting stack"
docker compose -f "$COMPOSE_FILE" up -d

echo "Done. Use 'docker compose -f $COMPOSE_FILE logs -f' to view logs."