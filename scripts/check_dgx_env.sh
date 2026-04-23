#!/usr/bin/env bash
# =============================================================================
# SmartVision-X — DGX Startup Validation
# =============================================================================
# Run this BEFORE docker compose up to verify the environment is ready.
# Usage: bash scripts/check_dgx_env.sh
# =============================================================================

set -e
PASS="✅"
FAIL="❌"
WARN="⚠️ "
INFO="→ "

errors=0

echo "=================================================="
echo "SmartVision-X — DGX Environment Pre-flight Check"
echo "=================================================="
echo ""

# ── Docker ─────────────────────────────────────────────────────────────────
echo "[ Docker ]"
if docker info &>/dev/null; then
    echo "  $PASS Docker daemon accessible"
else
    echo "  $FAIL Docker not accessible (try: sudo usermod -aG docker \$USER)"
    ((errors++))
fi

COMPOSE_VERSION=$(docker compose version 2>/dev/null | grep -oP '\d+\.\d+' | head -1)
if [ -n "$COMPOSE_VERSION" ]; then
    echo "  $PASS Docker Compose v$COMPOSE_VERSION"
else
    echo "  $FAIL Docker Compose v2 not found (try: docker compose version)"
    ((errors++))
fi

# ── GPU ────────────────────────────────────────────────────────────────────
echo ""
echo "[ GPU / NVIDIA ]"
if nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  $PASS nvidia-smi OK — $GPU_COUNT GPU(s) detected: $GPU_NAME"
else
    echo "  $FAIL nvidia-smi failed — GPU not available"
    ((errors++))
fi

if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &>/dev/null; then
    echo "  $PASS NVIDIA Container Toolkit working"
else
    echo "  $FAIL NVIDIA Container Toolkit not working"
    echo "      Fix: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    ((errors++))
fi

# ── Port conflicts ─────────────────────────────────────────────────────────
echo ""
echo "[ Port Availability ]"
PORTS=(8000 8001 8002 8003 9092 5432 6379 3000 9091 9094 9187 9121 9308)
for port in "${PORTS[@]}"; do
    if ss -tlnp "sport = :$port" 2>/dev/null | grep -q LISTEN; then
        # Check if it's one of our own containers vs something else
        PROC=$(ss -tlnp "sport = :$port" 2>/dev/null | grep LISTEN | awk '{print $NF}' | head -1)
        echo "  $WARN Port $port in use: $PROC"
    else
        echo "  $PASS Port $port free"
    fi
done

# ── Existing containers that we must NOT duplicate ─────────────────────────
echo ""
echo "[ Existing DGX Containers (must not conflict) ]"
for name in prometheus grafana; do
    if docker ps -q --filter "name=^${name}$" | grep -q .; then
        echo "  $INFO $name already running — will NOT be redeployed (correct)"
    else
        echo "  $INFO $name not found (OK — it may use a different name)"
    fi
done

for name in dcgm-exporter cadvisor; do
    if docker ps -q --filter "name=^${name}$" | grep -q .; then
        echo "  $INFO $name already running — GPU/container metrics already collected"
    fi
done

# ── Model files ────────────────────────────────────────────────────────────
echo ""
echo "[ Model Files ]"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DETECTION_MODEL="$PROJECT_ROOT/models/face_detection/1/model.onnx"
RECOGNITION_MODEL="$PROJECT_ROOT/models/face_recognition/1/model.onnx"

if [ -f "$DETECTION_MODEL" ]; then
    SIZE=$(du -h "$DETECTION_MODEL" | cut -f1)
    echo "  $PASS face_detection/1/model.onnx ($SIZE)"
else
    echo "  $FAIL models/face_detection/1/model.onnx missing"
    echo "      Fix: python download_model.py"
    ((errors++))
fi

if [ -f "$RECOGNITION_MODEL" ]; then
    SIZE=$(du -h "$RECOGNITION_MODEL" | cut -f1)
    echo "  $PASS face_recognition/1/model.onnx ($SIZE)"
else
    echo "  $FAIL models/face_recognition/1/model.onnx missing"
    echo "      Fix: python download_model.py"
    ((errors++))
fi

# ── RTSP camera ────────────────────────────────────────────────────────────
echo ""
echo "[ Camera Configuration ]"
if [ -n "$CAM_01_RTSP_URL" ]; then
    echo "  $PASS CAM_01_RTSP_URL set: $CAM_01_RTSP_URL"
else
    echo "  $WARN CAM_01_RTSP_URL not set — using default placeholder"
    echo "      Set: export CAM_01_RTSP_URL='rtsp://your-camera-ip:554/stream'"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
if [ "$errors" -eq 0 ]; then
    echo "✅  All checks passed — ready to run docker compose up"
    echo ""
    echo "Next steps:"
    echo "  cd infra/"
    echo "  docker compose up --build -d"
    echo "  bash scripts/patch_prometheus.sh   # wire to existing prometheus"
else
    echo "❌  $errors check(s) failed — fix above errors before proceeding"
fi
echo "=================================================="
exit $errors
