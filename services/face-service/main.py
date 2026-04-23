"""
SmartVision-X — Face Service  (Stage 1 + Stage 5)
===================================================
Changes from initial version:
  - Identity store migrated from face_db.json → PostgreSQL + pgvector
  - O(N) cosine scan replaced by HNSW ANN search (< 5ms at 1M identities)
  - Prometheus metrics instrumentation on all endpoints
  - Kafka event publishing on registration and recognition
  - /internal/search-embedding endpoint for inference-worker (Stage 3)
  - Startup/shutdown lifecycle hooks for DB pool

Endpoints:
  POST /register                   Register a face identity
  POST /recognize                  Identify face(s) in an image
  POST /internal/search-embedding  Internal ANN search for inference-worker
  GET  /identities                 List all registered identities
  DELETE /identities/{name}        Remove a registered identity
  GET  /cameras                    List registered cameras
  POST /cameras                    Register a new camera source
  GET  /health                     Service + Triton + DB health
  GET  /metrics                    Prometheus metrics endpoint
  GET  /api/docs                   OpenAPI documentation
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from kafka import KafkaProducer
from prometheus_client import (
    Counter, Gauge, Histogram,
    make_asgi_app, CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel
from starlette.responses import Response

from database import db
from face_detector import FaceDetector
from triton_client import infer_face_recognition

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("face-service")

# ── Prometheus metrics ─────────────────────────────────────────────────────────
REGISTRATIONS = Counter(
    "svx_registrations_total",
    "Total face registration attempts",
    ["status"],        # success | error_no_face | error_invalid | error_duplicate
)
RECOGNITIONS = Counter(
    "svx_recognitions_total",
    "Total recognition queries",
    ["result"],        # matched | unknown | error
)
RECOGNITION_LATENCY = Histogram(
    "svx_recognition_latency_seconds",
    "End-to-end /recognize latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)
REGISTER_LATENCY = Histogram(
    "svx_register_latency_seconds",
    "End-to-end /register latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)
TRITON_LATENCY = Histogram(
    "svx_triton_call_latency_seconds",
    "Triton inference call latency",
    ["model"],         # face_detection | face_recognition
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
REGISTERED_IDENTITIES = Gauge(
    "svx_registered_identities_total",
    "Current number of registered identities",
)
DB_SEARCH_LATENCY = Histogram(
    "svx_db_search_latency_seconds",
    "pgvector HNSW search latency",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1],
)

# ── Kafka producer ─────────────────────────────────────────────────────────────
KAFKA_BROKER   = os.environ.get("KAFKA_BROKER", "kafka:9092")
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.45"))

_kafka_producer: KafkaProducer | None = None

def get_kafka_producer() -> KafkaProducer | None:
    global _kafka_producer
    if _kafka_producer is not None:
        return _kafka_producer
    try:
        _kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode(),
            key_serializer=str.encode,
            acks=1,
            request_timeout_ms=3000,
            retries=2,
        )
        logger.info("Kafka producer connected to %s", KAFKA_BROKER)
    except Exception as exc:
        logger.warning("Kafka unavailable — events will be skipped: %s", exc)
    return _kafka_producer


def publish_event(topic: str, key: str, payload: dict) -> None:
    """Publish a Kafka event. Non-blocking; failures are logged and swallowed."""
    try:
        producer = get_kafka_producer()
        if producer:
            producer.send(topic=topic, value=payload, key=key)
    except Exception as exc:
        logger.debug("Kafka publish failed [%s]: %s", topic, exc)


# ── Application lifecycle ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SmartVision-X face-service…")
    await db.init()
    count = await db.count_identities()
    REGISTERED_IDENTITIES.set(count)
    logger.info("Startup complete — %d identities registered", count)
    yield
    # Shutdown
    await db.close()
    logger.info("Database pool closed")


app = FastAPI(
    title="SmartVision-X Face Service",
    version="2.0.0",
    description="GPU-accelerated face registration and recognition service",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Serve GUI
GUI_DIR = os.path.join(os.path.dirname(__file__), "gui")
if os.path.isdir(GUI_DIR):
    app.mount("/static", StaticFiles(directory=GUI_DIR), name="static")

detector = FaceDetector()


# ── Image helpers ──────────────────────────────────────────────────────────────
def decode_image(contents: bytes) -> np.ndarray:
    if not contents:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    np_img = np.frombuffer(contents, dtype=np.uint8)
    img    = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. Ensure the file is a valid JPEG or PNG.",
        )
    if img.ndim != 3 or img.shape[2] != 3:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 3-channel BGR image, got shape {img.shape}",
        )
    if img.shape[0] < 32 or img.shape[1] < 32:
        raise HTTPException(
            status_code=422,
            detail="Image too small — minimum 32×32 pixels required",
        )
    return img


def preprocess_face(face: np.ndarray) -> np.ndarray:
    """Resize to 112×112, CHW float32 in [0,1]."""
    face = cv2.resize(face, (112, 112))
    face = face.astype(np.float32) / 255.0
    face = face.transpose(2, 0, 1)
    return np.expand_dims(face, axis=0)


def normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


def extract_embedding(img: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = img[y1:y2, x1:x2]

    if face.size == 0 or (y2 - y1) < 10 or (x2 - x1) < 10:
        raise HTTPException(
            status_code=400,
            detail="Face crop too small — try a clearer, larger image",
        )

    face_tensor = preprocess_face(face)

    t0 = time.perf_counter()
    try:
        embedding = infer_face_recognition(face_tensor).flatten()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Recognition model unavailable: {exc}",
        )
    TRITON_LATENCY.labels(model="face_recognition").observe(
        time.perf_counter() - t0
    )

    return normalise(embedding)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_gui():
    index = os.path.join(GUI_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"service": "SmartVision-X", "docs": "/api/docs"})


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    checks: dict = {}

    # Triton
    try:
        r = requests.get("http://triton:8000/v2/health/ready", timeout=3)
        checks["triton"] = "ready" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as exc:
        checks["triton"] = f"error: {str(exc)[:60]}"

    # Triton models
    for model in ("face_detection", "face_recognition"):
        try:
            r = requests.get(f"http://triton:8000/v2/models/{model}/ready", timeout=3)
            checks[model] = "ready" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception:
            checks[model] = "unreachable"

    # PostgreSQL
    try:
        count = await db.count_identities()
        checks["postgres"] = f"ready ({count} identities)"
        REGISTERED_IDENTITIES.set(count)
    except Exception as exc:
        checks["postgres"] = f"error: {str(exc)[:60]}"

    all_ready = all(
        v.startswith("ready") for v in checks.values()
    )

    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "service": "face-service",
            "version": "2.0.0",
            "status":  "healthy" if all_ready else "degraded",
            "checks":  checks,
        },
    )


# ── Register ───────────────────────────────────────────────────────────────────
@app.post("/register", tags=["Biometric"])
async def register_face(name: str, file: UploadFile = File(...)):
    """
    Register a named face identity.
    Stores a 512-d ArcFace embedding in PostgreSQL + pgvector.
    """
    t0 = time.perf_counter()

    if not name or not name.strip():
        REGISTRATIONS.labels(status="error_invalid").inc()
        raise HTTPException(status_code=422, detail="Parameter 'name' must not be empty")

    name     = name.strip()
    contents = await file.read()
    img      = decode_image(contents)

    # Detect
    t_det = time.perf_counter()
    faces = detector.detect(img)
    TRITON_LATENCY.labels(model="face_detection").observe(time.perf_counter() - t_det)

    if not faces:
        REGISTRATIONS.labels(status="error_no_face").inc()
        raise HTTPException(
            status_code=400,
            detail="No face detected. Ensure the face is clearly visible and well-lit.",
        )

    # Check duplicate
    if await db.identity_exists(name):
        REGISTRATIONS.labels(status="error_duplicate").inc()
        raise HTTPException(
            status_code=409,
            detail=f"Identity '{name}' is already registered. "
                   "Delete it first or choose a different name.",
        )

    # Extract embedding
    bbox      = faces[0]
    embedding = extract_embedding(img, bbox)

    # Persist to pgvector
    row = await db.register_identity(name, embedding, source="manual")

    # Update Prometheus gauge
    count = await db.count_identities()
    REGISTERED_IDENTITIES.set(count)

    # Publish Kafka event
    publish_event(
        topic="face.events",
        key="register",
        payload={
            "event":    "face_registered",
            "name":     name,
            "id":       row["id"],
            "source":   "api",
        },
    )

    REGISTRATIONS.labels(status="success").inc()
    REGISTER_LATENCY.observe(time.perf_counter() - t0)

    x1, y1, x2, y2 = bbox
    return {
        "status":            "registered",
        "name":              name,
        "id":                row["id"],
        "faces_detected":    len(faces),
        "face_used":         {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "total_registered":  count,
    }


# ── Recognize ──────────────────────────────────────────────────────────────────
@app.post("/recognize", tags=["Biometric"])
async def recognize_face(file: UploadFile = File(...)):
    """
    Identify face(s) in an image against the pgvector identity store.
    Uses HNSW ANN search — target < 10ms per query at 1M identities.
    """
    t0 = time.perf_counter()

    contents = await file.read()
    img      = decode_image(contents)

    t_det = time.perf_counter()
    faces = detector.detect(img)
    TRITON_LATENCY.labels(model="face_detection").observe(time.perf_counter() - t_det)

    if not faces:
        return {"faces": [], "message": "No face detected in the uploaded image"}

    if await db.count_identities() == 0:
        raise HTTPException(
            status_code=400,
            detail="No identities registered. Register at least one face first.",
        )

    results = []

    for bbox in faces:
        try:
            embedding = extract_embedding(img, bbox)
        except HTTPException:
            continue

        # ANN search via pgvector HNSW
        t_search = time.perf_counter()
        matches  = await db.search_identity(embedding, threshold=MATCH_THRESHOLD)
        DB_SEARCH_LATENCY.observe(time.perf_counter() - t_search)

        x1, y1, x2, y2 = bbox

        if matches:
            best = matches[0]
            result = {
                "person":     best["name"],
                "confidence": round(best["similarity"], 4),
                "matched":    True,
                "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
            RECOGNITIONS.labels(result="matched").inc()
        else:
            result = {
                "person":     "unknown",
                "confidence": 0.0,
                "matched":    False,
                "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
            RECOGNITIONS.labels(result="unknown").inc()
            # Store for future clustering (Stage 6)
            await db.store_candidate_face(embedding)

        results.append(result)

        # Log event to PostgreSQL
        await db.log_recognition_event(
            person_name=result["person"],
            confidence=result["confidence"],
            matched=result["matched"],
            bbox=result["box"],
            source="api",
        )

        # Publish Kafka event
        publish_event(
            topic="face.events",
            key="recognize",
            payload={
                "event":      "face_recognized",
                "person":     result["person"],
                "confidence": result["confidence"],
                "matched":    result["matched"],
                "source":     "api",
            },
        )

    RECOGNITION_LATENCY.observe(time.perf_counter() - t0)

    return {"faces": results, "total_detected": len(faces)}


# ── Internal: embedding search (used by inference-worker) ─────────────────────
class EmbeddingSearchRequest(BaseModel):
    embedding: list[float]
    threshold: float = 0.45
    top_k:     int   = 1
    camera_id: str | None = None
    track_id:  int | None = None


@app.post("/internal/search-embedding", include_in_schema=False)
async def search_embedding(req: EmbeddingSearchRequest):
    """
    Internal endpoint for inference-worker to search pgvector.
    Not exposed in public API documentation.
    """
    emb     = normalise(np.array(req.embedding, dtype=np.float32))
    t0      = time.perf_counter()
    matches = await db.search_identity(emb, threshold=req.threshold, top_k=req.top_k)
    DB_SEARCH_LATENCY.observe(time.perf_counter() - t0)

    if matches:
        best = matches[0]
        result = {
            "person":     best["name"],
            "confidence": round(best["similarity"], 4),
            "matched":    True,
        }
        RECOGNITIONS.labels(result="matched").inc()
    else:
        result = {
            "person":     "unknown",
            "confidence": 0.0,
            "matched":    False,
        }
        RECOGNITIONS.labels(result="unknown").inc()
        await db.store_candidate_face(emb, req.camera_id, req.track_id)

    # Log event
    await db.log_recognition_event(
        person_name=result["person"],
        confidence=result["confidence"],
        matched=result["matched"],
        camera_id=req.camera_id,
        track_id=req.track_id,
        source="stream",
    )

    return result


# ── Identity management ────────────────────────────────────────────────────────
@app.get("/identities", tags=["Biometric"])
async def list_identities():
    identities = await db.list_identities()
    return {
        "total":      len(identities),
        "identities": identities,
    }


@app.delete("/identities/{name}", tags=["Biometric"])
async def delete_identity(name: str):
    deleted = await db.delete_identity(name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Identity '{name}' not found.",
        )
    count = await db.count_identities()
    REGISTERED_IDENTITIES.set(count)
    publish_event(
        topic="face.events",
        key="delete",
        payload={"event": "identity_deleted", "name": name},
    )
    return {"status": "deleted", "name": name, "remaining": count}


# ── Camera management ──────────────────────────────────────────────────────────
class CameraRegistration(BaseModel):
    camera_id:    str
    name:         str = ""
    rtsp_url:     str | None = None
    location:     str | None = None
    frame_sample: int = 3


@app.post("/cameras", tags=["Cameras"])
async def register_camera(req: CameraRegistration):
    await db.register_camera(
        camera_id=req.camera_id,
        name=req.name,
        rtsp_url=req.rtsp_url,
        location=req.location,
        frame_sample=req.frame_sample,
    )
    return {"status": "registered", "camera_id": req.camera_id}


@app.get("/cameras", tags=["Cameras"])
async def list_cameras():
    cameras = await db.list_cameras()
    return {"total": len(cameras), "cameras": cameras}


# ── Debug ──────────────────────────────────────────────────────────────────────
@app.get("/debug/db", tags=["Debug"])
async def debug_db():
    """Summary of identity store — no embeddings returned."""
    count = await db.count_identities()
    return {
        "total": count,
        "store": "postgresql+pgvector",
        "note":  "Use GET /identities for full list",
    }
