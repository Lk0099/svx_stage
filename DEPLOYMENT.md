# SmartVision-X — Deployment Guide
## Stages 1 · 2 · 3 · 5 Implementation

---

## What Is Implemented in This Build

| Stage | Feature | Status |
|-------|---------|--------|
| Stage 1 | pgvector identity store — replaces face_db.json | ✅ Complete |
| Stage 5 | Prometheus + Grafana observability | ✅ Complete |
| Stage 2 | RTSP camera stream ingest → Kafka | ✅ Complete |
| Stage 3 | ByteTrack tracking + Triton gRPC inference | ✅ Complete |

---

## New Files in This Build

```
services/
  face-service/
    database.py          ← STAGE 1: async pgvector client (asyncpg + HNSW search)
    main.py              ← STAGE 1+5: pgvector queries + Prometheus metrics + Kafka events
  ingest-service/        ← STAGE 2: NEW — RTSP camera frame producer
    ingest.py
    Dockerfile
    requirements.txt
  inference-worker/      ← STAGE 3: NEW — ByteTrack + Triton gRPC + pgvector search
    worker.py
    Dockerfile
    requirements.txt
infra/
  init.sql               ← STAGE 1: PostgreSQL schema with pgvector + HNSW index
  docker-compose.yml     ← ALL: Updated with all 4 stages wired
  prometheus/
    prometheus.yml        ← STAGE 5: Scrape config for all services
  grafana/
    dashboards/
      smartvision_main.json  ← STAGE 5: Pre-built Grafana dashboard
    provisioning/...         ← STAGE 5: Auto-provisioning config
scripts/
  migrate_to_postgres.py    ← STAGE 1: Migrate existing face_db.json to pgvector
  test_integration.py       ← ALL: Integration test suite
```

---

## Prerequisites

- Docker Engine ≥ 24 with Docker Compose v2
- NVIDIA Container Toolkit installed
- DGX system with CUDA-capable GPUs
- Python 3.10+ (for scripts only)
- RTSP camera streams available on the network

---

## Step-by-Step Deployment

### Step 1 — Download model weights

```bash
cd smart-vision-x/
python download_model.py
```

Verify:
```bash
ls -lh models/face_detection/1/model.onnx    # SCRFD detection
ls -lh models/face_recognition/1/model.onnx  # ArcFace recognition
```

---

### Step 2 — Configure cameras

Edit `infra/docker-compose.yml` and set your camera RTSP URLs:

```yaml
ingest-cam-01:
  environment:
    CAMERA_ID: "cam-entrance-01"
    RTSP_URL:  "rtsp://192.168.1.101:554/stream1"
    FRAME_SAMPLE: "3"      # process every 3rd frame (30fps → 10fps)
```

Add one `ingest-cam-XX` block per camera. Or use environment variables:

```bash
export CAM_01_RTSP_URL="rtsp://192.168.1.101:554/stream1"
```

---

### Step 3 — Start all services

```bash
cd infra/
docker compose up --build -d
```

Watch startup:
```bash
docker compose logs -f triton face-service postgres
```

Wait for Triton to report both models `READY`:
```
I triton-service | model 'face_detection'  READY
I triton-service | model 'face_recognition' READY
```

---

### Step 4 — Migrate existing registrations (if upgrading)

If you have existing data in `face_db.json`:

```bash
# From project root
pip install asyncpg numpy
export DATABASE_URL="postgresql://admin:admin@localhost:5432/smartvision"
python scripts/migrate_to_postgres.py
```

---

### Step 5 — Verify Triton models

```bash
curl http://localhost:8000/v2/models/face_detection/ready
curl http://localhost:8000/v2/models/face_recognition/ready
# Both should return HTTP 200
```

---

### Step 6 — Verify face-service (Stage 1 + 5)

```bash
# Health check — should show postgres: "ready (N identities)"
curl http://localhost:8003/health | python3 -m json.tool

# Prometheus metrics
curl http://localhost:8003/metrics | grep svx_

# Register a face
curl -X POST "http://localhost:8003/register?name=Alice" \
     -F "file=@test_images/IMG_20230806_224456_048.jpg"

# List identities (from pgvector)
curl http://localhost:8003/identities
```

---

### Step 7 — Open Grafana dashboard (Stage 5)

Navigate to: **http://localhost:3001**
- Username: `admin`
- Password: `admin`

The **SmartVision-X Operations** dashboard is pre-provisioned.

Key panels:
- GPU Utilization — Triton GPU usage %
- Recognition Latency p99 — alert if > 200ms
- Unknown Face Rate — alert if > 80%
- Kafka Consumer Lag — alert if > 10,000 messages
- pgvector Search Latency p95 — target < 10ms

---

### Step 8 — Verify streaming pipeline (Stage 2 + 3)

```bash
# Check ingest worker logs
docker compose logs -f ingest-cam-01

# Check inference worker logs
docker compose logs -f inference-worker

# Watch recognition events on Kafka
docker compose exec kafka \
  kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic camera.cam-01.events \
  --from-beginning
```

---

### Step 9 — Scale inference workers

For more cameras or higher throughput:

```bash
# Scale to 4 parallel inference workers
docker compose up --scale inference-worker=4 -d
```

Each worker picks up camera topics via Kafka consumer group load balancing.

---

### Step 10 — Run integration tests

```bash
pip install requests Pillow
python scripts/test_integration.py
# Or test a specific stage:
python scripts/test_integration.py --test stage1
python scripts/test_integration.py --test stage5
```

---

## Service Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| face-service | 8003 | REST API + Prometheus metrics |
| triton HTTP | 8000 | Inference HTTP API |
| triton gRPC | 8001 | Inference gRPC (used by worker) |
| triton metrics | 8002 | Prometheus scrape endpoint |
| gui-service | 3000 | Web dashboard |
| grafana | 3001 | Observability dashboard |
| prometheus | 9090 | Metrics server |
| postgres | 5432 | pgvector database |
| kafka | 9092 | Message broker |
| redis | 6379 | Track state cache |

---

## Key API Endpoints

### Face Service (http://localhost:8003)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/register?name=X` | Register a face identity |
| POST | `/recognize` | Identify face(s) in image |
| GET | `/identities` | List all registered identities |
| DELETE | `/identities/{name}` | Remove an identity |
| POST | `/cameras` | Register a camera source |
| GET | `/cameras` | List registered cameras |
| GET | `/health` | Full system health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/api/docs` | Swagger documentation |

### Internal (used by inference-worker)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/internal/search-embedding` | ANN search by embedding vector |

---

## Troubleshooting

**Triton models not loading:**
```bash
docker compose logs triton | grep -E "ERROR|FAILED|READY"
# Check model files exist:
ls models/face_detection/1/model.onnx
ls models/face_recognition/1/model.onnx
```

**PostgreSQL connection refused:**
```bash
docker compose ps postgres
docker compose logs postgres | tail -20
# Wait for "database system is ready to accept connections"
```

**Ingest worker not publishing:**
```bash
docker compose logs ingest-cam-01 | tail -20
# Check RTSP URL is reachable:
docker compose exec ingest-cam-01 python3 -c \
  "import cv2; cap=cv2.VideoCapture('$RTSP_URL'); print('open:', cap.isOpened())"
```

**High unknown face rate (> 80%):**
- Register more identities via `/register`
- Lower `MATCH_THRESHOLD` to 0.40 (more permissive)
- Check face image quality — ensure faces are ≥ 80×80 pixels

**Kafka consumer lag growing:**
```bash
# Scale up inference workers
docker compose up --scale inference-worker=4 -d
# Monitor lag in Grafana panel "Kafka Consumer Lag"
```

---

## Architecture Summary

```
RTSP Camera N
    │
    ▼
[ingest-service]           Stage 2
  RTSP → JPEG → Kafka
  topic: camera.{id}.frames
    │
    ▼
[Kafka]
    │
    ▼
[inference-worker]         Stage 3
  ByteTrack (Redis)
  SCRFD → Triton gRPC      (no base64 overhead)
  ArcFace → Triton gRPC    (only on new tracks)
  pgvector search          (via face-service)
  topic: camera.{id}.events
    │
    ├─→ [alert-service]    (Kafka consumer)
    ├─→ [tracking-service] (Kafka consumer)
    └─→ [Grafana]          Stage 5 (via Prometheus)

[face-service]             Stage 1 + 5
  POST /register → pgvector (HNSW index)
  POST /recognize → pgvector ANN search
  GET  /metrics   → Prometheus
```
