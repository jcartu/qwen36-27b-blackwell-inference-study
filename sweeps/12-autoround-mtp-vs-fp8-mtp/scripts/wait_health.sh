#!/usr/bin/env bash
# Wait for vllm-exp12-autoround-mtp /health 200, with hard cap and fail detection.
set -uo pipefail
CONTAINER="${1:-vllm-exp12-autoround-mtp}"
PORT="${2:-8765}"
LIMIT="${3:-600}"

echo "Waiting for ${CONTAINER} /health on :${PORT} (max ${LIMIT}s)..."
SECS=0
while (( SECS < LIMIT )); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[FAIL @${SECS}s] Container exited"
    docker logs --tail 60 "$CONTAINER" 2>&1
    exit 1
  fi
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[OK @${SECS}s] /health 200"
    exit 0
  fi
  if docker logs "$CONTAINER" 2>&1 | grep -qE "EngineCore failed|NotImplementedError|ValueError.*spec|raise RuntimeError"; then
    echo "[FAIL @${SECS}s] Detected fatal error"
    docker logs "$CONTAINER" 2>&1 | grep -E "Error|Exception" | tail -20
    exit 2
  fi
  sleep 10
  SECS=$((SECS + 10))
done
echo "[TIMEOUT] No /health 200 after ${LIMIT}s"
docker logs --tail 60 "$CONTAINER" 2>&1
exit 3
