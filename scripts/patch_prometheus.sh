#!/usr/bin/env bash
# =============================================================================
# SmartVision-X — Patch Existing DGX Prometheus
# =============================================================================
# This script adds SmartVision-X scrape targets to the Prometheus instance
# already running on your DGX system.
#
# HOW IT WORKS:
#   1. Finds the existing prometheus container's config file
#   2. Appends SmartVision-X scrape job blocks
#   3. Sends SIGHUP (or HTTP reload) to prometheus to hot-reload config
#
# USAGE:
#   bash scripts/patch_prometheus.sh
#
# PREREQUISITES:
#   - Docker must be accessible (you may need to run as root or with sudo)
#   - Existing prometheus container must be named 'prometheus'
# =============================================================================

set -e

PROMETHEUS_CONTAINER="prometheus"
SVX_CONFIG_BLOCK="/tmp/svx_prometheus_targets.yml"
HOST_IP=$(hostname -I | awk '{print $1}')

echo "=== SmartVision-X Prometheus Patch ==="
echo "DGX host IP detected: $HOST_IP"
echo ""

# ── Step 1: Find prometheus config file ────────────────────────────────────
echo "[1/4] Locating prometheus config inside container..."
CONFIG_PATH=$(docker exec "$PROMETHEUS_CONTAINER" \
    sh -c 'ps aux | grep prometheus | grep -o "\-\-config\.file=[^ ]*" | cut -d= -f2' 2>/dev/null \
    || echo "/etc/prometheus/prometheus.yml")

echo "      Config path: $CONFIG_PATH"

# ── Step 2: Check if SVX targets already added ─────────────────────────────
echo "[2/4] Checking if SmartVision-X targets already present..."
if docker exec "$PROMETHEUS_CONTAINER" grep -q "svx-face-service" "$CONFIG_PATH" 2>/dev/null; then
    echo "      SmartVision-X targets already present in prometheus config."
    echo "      Sending reload signal only..."
    docker exec "$PROMETHEUS_CONTAINER" kill -HUP 1 2>/dev/null || \
        curl -s -X POST http://localhost:9090/-/reload
    echo "      Prometheus reloaded."
    exit 0
fi

# ── Step 3: Write the SmartVision-X scrape config block ────────────────────
echo "[3/4] Writing SmartVision-X scrape targets..."

cat > "$SVX_CONFIG_BLOCK" << SCRAPEEOF

  # ── SmartVision-X Services (added by patch_prometheus.sh) ──────────────────

  - job_name: svx-face-service
    static_configs:
      - targets: ['${HOST_IP}:8003']
        labels:
          service: 'svx-face-service'
          system:  'smartvision-x'
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: svx-triton
    static_configs:
      - targets: ['${HOST_IP}:8002']
        labels:
          service: 'svx-triton'
          system:  'smartvision-x'
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: svx-inference-worker
    static_configs:
      - targets: ['${HOST_IP}:9094']
        labels:
          service: 'svx-inference-worker'
          system:  'smartvision-x'
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: svx-ingest-cam01
    static_configs:
      - targets: ['${HOST_IP}:9091']
        labels:
          service:   'svx-ingest'
          camera_id: 'cam-01'
          system:    'smartvision-x'
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: svx-postgres-exporter
    static_configs:
      - targets: ['${HOST_IP}:9187']
        labels:
          service: 'svx-postgres'
          system:  'smartvision-x'
    scrape_interval: 30s

  - job_name: svx-redis-exporter
    static_configs:
      - targets: ['${HOST_IP}:9121']
        labels:
          service: 'svx-redis'
          system:  'smartvision-x'
    scrape_interval: 30s

  - job_name: svx-kafka-exporter
    static_configs:
      - targets: ['${HOST_IP}:9308']
        labels:
          service: 'svx-kafka'
          system:  'smartvision-x'
    scrape_interval: 15s
SCRAPEEOF

echo "      Scrape config block written to $SVX_CONFIG_BLOCK"

# ── Step 4: Inject config and reload ───────────────────────────────────────
echo "[4/4] Injecting config into prometheus container..."

# Copy the block into the container
docker cp "$SVX_CONFIG_BLOCK" \
    "$PROMETHEUS_CONTAINER":/tmp/svx_targets.yml

# Append the block to the prometheus config (if not already present)
docker exec "$PROMETHEUS_CONTAINER" sh -c \
    "cat /tmp/svx_targets.yml >> $CONFIG_PATH"

echo "      Config updated. Sending reload signal..."

# Try SIGHUP first, then HTTP reload endpoint
docker exec "$PROMETHEUS_CONTAINER" kill -HUP 1 2>/dev/null && \
    echo "      Prometheus reloaded via SIGHUP" || \
    (curl -s -X POST http://localhost:9090/-/reload && \
     echo "      Prometheus reloaded via HTTP API")

echo ""
echo "=== Done ==="
echo ""
echo "Verify targets at: http://localhost:9090/targets"
echo "Filter by: system='smartvision-x'"
echo ""
echo "Expected targets:"
echo "  svx-face-service      → http://${HOST_IP}:8003/metrics"
echo "  svx-triton            → http://${HOST_IP}:8002/metrics"
echo "  svx-inference-worker  → http://${HOST_IP}:9094/metrics"
echo "  svx-ingest-cam01      → http://${HOST_IP}:9091/metrics"
echo "  svx-postgres-exporter → http://${HOST_IP}:9187/metrics"
echo "  svx-redis-exporter    → http://${HOST_IP}:9121/metrics"
echo "  svx-kafka-exporter    → http://${HOST_IP}:9308/metrics"
