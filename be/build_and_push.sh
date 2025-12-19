#!/usr/bin/env bash
set -euo pipefail

# This script is a shim to the project root build_and_push.sh. Please run the
# top-level script instead. This file exists for backwards compatibility.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_SCRIPT="$(cd "${SCRIPT_DIR}/.." && pwd)/build_and_push.sh"
if [[ ! -f "$ROOT_SCRIPT" ]]; then
  echo "Error: root build_and_push.sh not found at $ROOT_SCRIPT"
  exit 1
fi
# Forward all args to the root script
exec "$ROOT_SCRIPT" "$@"
