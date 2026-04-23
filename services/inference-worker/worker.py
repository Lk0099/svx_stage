"""
SmartVision-X — Inference Worker  (Stage 2 + Stage 3)
======================================================
Consumes camera frame topics from Kafka, runs GPU inference via Triton
(gRPC binary protocol — no base64 overhead), and maintains per-camera
ByteTrack identity continuity.

Architecture:
    Kafka: camera.*.frames
        → SCRFD detection (Triton gRPC)
        → ByteTrack (per-camera, Redis-backed state)
        → ArcFace recognition ONLY on new/unconfirmed tracks
        → pgvector search (via face-service /internal/search-embedding)
        → Publish results to Kafka: camera.*.events

ByteTrack reduces ArcFace recognition calls by ~85% at 30fps by reusing
the identity from previous frames for confirmed tracks.

Environment variables:
    TRITON_URL          Triton gRPC endpoint. Default: triton:8001
    KAFKA_BROKER        Kafka broker. Default: kafka:9092
    REDIS_HOST          Redis host. Default: redis
    FACE_SERVICE_URL    face-service base URL. Default: http://face-service:8003
    WORKER_ID           Unique worker identifier. Default: worker-01
    MATCH_THRESHOLD     Recognition threshold. Default: 0.45
    RECONFIRM_INTERVAL  Frames between re-recognition of unconfirmed tracks. Default: 30
    TRACK_TTL_SECONDS   Redis TTL for track state. Default: 30
    METRICS_PORT        Prometheus metrics port. Default: 9090
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import redis
import requests
import tritonclient.grpc as grpcclient
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ByteTracker — lightweight Python port of ByteTrack
# Install: pip install bytetracker
try:
    from bytetracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    logging.warning("bytetracker not installed — tracking disabled, full inference per frame")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("worker")

# ── Config ─────────────────────────────────────────────────────────────────────
TRITON_URL          = os.environ.get("TRITON_URL",         "triton:8001")
KAFKA_BROKER        = os.environ.get("KAFKA_BROKER",       "kafka:9092")
REDIS_HOST          = os.environ.get("REDIS_HOST",         "redis")
FACE_SERVICE_URL    = os.environ.get("FACE_SERVICE_URL",   "http://face-service:8003")
WORKER_ID           = os.environ.get("WORKER_ID",          "worker-01")
MATCH_THRESHOLD     = float(os.environ.get("MATCH_THRESHOLD",    "0.45"))
RECONFIRM_INTERVAL  = int(os.environ.get("RECONFIRM_INTERVAL",   "30"))
TRACK_TTL_SECONDS   = int(os.environ.get("TRACK_TTL_SECONDS",    "30"))
METRICS_PORT        = int(os.environ.get("METRICS_PORT",         "9090"))

# ── Prometheus metrics ─────────────────────────────────────────────────────────
FRAMES_PROCESSED = Counter(
    "svx_frames_processed_total",
    "Frames consumed from Kafka",
    ["camera_id"],
)
FACES_DETECTED = Counter(
    "svx_faces_detected_total",
    "Total SCRFD face detections",
    ["camera_id"],
)
RECOGNITION_CALLS = Counter(
    "svx_recognition_calls_total",
    "ArcFace inference calls (after track filtering)",
    ["camera_id"],
)
RECOGNITION_RESULTS = Counter(
    "svx_worker_recognitions_total",
    "Recognition results by outcome",
    ["camera_id", "result"],
)
DETECTION_LATENCY = Histogram(
    "svx_worker_detection_latency_seconds",
    "SCRFD Triton gRPC call latency",
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
)
RECOGNITION_LATENCY_W = Histogram(
    "svx_worker_recognition_latency_seconds",
    "ArcFace Triton gRPC call latency",
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1],
)
SEARCH_LATENCY = Histogram(
    "svx_worker_search_latency_seconds",
    "pgvector search call latency (via face-service)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05],
)
ACTIVE_TRACKS = Gauge(
    "svx_active_tracks",
    "Currently tracked faces per camera",
    ["camera_id"],
)
BYTETRACK_SKIP_RATIO = Gauge(
    "svx_bytetrack_skip_ratio",
    "Fraction of tracked faces that skipped ArcFace (0-1)",
    ["camera_id"],
)


# ── Prometheus HTTP server ─────────────────────────────────────────────────────
class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            data = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(data)
        elif self.path in ("/health", "/"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def start_metrics_server():
    server = HTTPServer(("0.0.0.0", METRICS_PORT), MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Metrics server on port %d", METRICS_PORT)


# ── Triton gRPC client ─────────────────────────────────────────────────────────
_triton: grpcclient.InferenceServerClient | None = None


def get_triton() -> grpcclient.InferenceServerClient:
    global _triton
    if _triton is not None:
        return _triton
    while True:
        try:
            client = grpcclient.InferenceServerClient(
                url=TRITON_URL, verbose=False
            )
            client.is_server_ready()
            _triton = client
            logger.info("Triton gRPC connected at %s", TRITON_URL)
            return _triton
        except Exception as exc:
            logger.warning("Triton not ready (%s) — retry in 5s", exc)
            time.sleep(5)


def run_detection(image_bytes: bytes) -> np.ndarray:
    """
    Send raw JPEG bytes to SCRFD via Triton gRPC (binary protocol, no base64).
    Returns ndarray (N, 5): [x1, y1, x2, y2, score]
    """
    img_array = np.array([image_bytes], dtype=object)
    inp = grpcclient.InferInput("IMAGE", [1], "BYTES")
    inp.set_data_from_numpy(img_array)
    out = grpcclient.InferRequestedOutput("BBOXES")

    t0 = time.perf_counter()
    try:
        resp = get_triton().infer(
            model_name="face_detection",
            inputs=[inp],
            outputs=[out],
        )
    except Exception as exc:
        logger.error("SCRFD inference failed: %s", exc)
        return np.zeros((0, 5), dtype=np.float32)
    finally:
        DETECTION_LATENCY.observe(time.perf_counter() - t0)

    boxes = resp.as_numpy("BBOXES")
    return boxes.reshape(-1, 5) if boxes.size > 0 else np.zeros((0, 5), dtype=np.float32)


def run_recognition(face_crop: np.ndarray) -> np.ndarray:
    """
    Send 112×112 BGR crop to ArcFace via Triton gRPC.
    Returns L2-normalised 512-d embedding.
    """
    face = cv2.resize(face_crop, (112, 112)).astype(np.float32) / 255.0
    face = face.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 112, 112)

    inp = grpcclient.InferInput("input.1", list(face.shape), "FP32")
    inp.set_data_from_numpy(face)
    out = grpcclient.InferRequestedOutput("683")

    t0 = time.perf_counter()
    try:
        resp = get_triton().infer(
            model_name="face_recognition",
            inputs=[inp],
            outputs=[out],
        )
    except Exception as exc:
        logger.error("ArcFace inference failed: %s", exc)
        return np.zeros(512, dtype=np.float32)
    finally:
        RECOGNITION_LATENCY_W.observe(time.perf_counter() - t0)

    emb = resp.as_numpy("683").flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-9 else emb


# ── pgvector search via face-service ──────────────────────────────────────────
def search_embedding(
    embedding:  np.ndarray,
    camera_id:  str | None = None,
    track_id:   int | None = None,
) -> dict:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{FACE_SERVICE_URL}/internal/search-embedding",
            json={
                "embedding":  embedding.tolist(),
                "threshold":  MATCH_THRESHOLD,
                "camera_id":  camera_id,
                "track_id":   track_id,
            },
            timeout=5,
        )
        SEARCH_LATENCY.observe(time.perf_counter() - t0)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.warning("Search request failed: %s", exc)
    return {"person": "unknown", "confidence": 0.0, "matched": False}


# ── Redis track state ──────────────────────────────────────────────────────────
_redis: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis
    if _redis is not None:
        return _redis
    while True:
        try:
            r = redis.Redis(
                host=REDIS_HOST, port=6379,
                decode_responses=False,
                socket_timeout=3,
            )
            r.ping()
            _redis = r
            logger.info("Redis connected at %s", REDIS_HOST)
            return _redis
        except Exception as exc:
            logger.warning("Redis not ready (%s) — retry in 3s", exc)
            time.sleep(3)


def get_track_state(camera_id: str, track_id: int) -> dict | None:
    key = f"svx:track:{camera_id}:{track_id}"
    try:
        data = get_redis().get(key)
        return json.loads(data) if data else None
    except Exception:
        return None


def set_track_state(camera_id: str, track_id: int, state: dict) -> None:
    key = f"svx:track:{camera_id}:{track_id}"
    try:
        get_redis().setex(key, TRACK_TTL_SECONDS, json.dumps(state))
    except Exception as exc:
        logger.debug("Redis write failed: %s", exc)


def get_and_increment_frame_count(camera_id: str, track_id: int) -> int:
    key = f"svx:trackfc:{camera_id}:{track_id}"
    try:
        r = get_redis()
        val = r.incr(key)
        r.expire(key, TRACK_TTL_SECONDS)
        return int(val)
    except Exception:
        return 1


# ── ByteTrack instances (per-camera, in-process) ──────────────────────────────
_trackers: dict[str, "BYTETracker"] = {}


def get_tracker(camera_id: str) -> "BYTETracker | None":
    if not BYTETRACK_AVAILABLE:
        return None
    if camera_id not in _trackers:
        _trackers[camera_id] = BYTETracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=10,
        )
        logger.info("ByteTracker initialised for camera %s", camera_id)
    return _trackers[camera_id]


# ── Kafka ──────────────────────────────────────────────────────────────────────
def make_consumer() -> KafkaConsumer:
    while True:
        try:
            c = KafkaConsumer(
                bootstrap_servers=KAFKA_BROKER,
                group_id="inference-workers",
                auto_offset_reset="latest",     # drop stale frames on restart
                enable_auto_commit=True,
                max_poll_records=5,             # process frames in small batches
                fetch_max_bytes=10_485_760,
                value_deserializer=None,        # raw bytes
            )
            c.subscribe(pattern=r"camera\..*\.frames")
            logger.info("Kafka consumer subscribed to camera.*.frames")
            return c
        except Exception as exc:
            logger.warning("Kafka unavailable (%s) — retry in 5s", exc)
            time.sleep(5)


def make_producer() -> KafkaProducer:
    while True:
        try:
            p = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode(),
                key_serializer=str.encode,
                acks=1,
            )
            logger.info("Kafka producer ready")
            return p
        except Exception as exc:
            logger.warning("Kafka producer unavailable (%s) — retry in 5s", exc)
            time.sleep(5)


# ── Frame processing ───────────────────────────────────────────────────────────
def process_frame(
    producer:    KafkaProducer,
    camera_id:   str,
    frame_bytes: bytes,
    frame_seq:   int,
    timestamp:   str,
) -> None:
    FRAMES_PROCESSED.labels(camera_id=camera_id).inc()

    # Decode frame for crop extraction (detection uses raw bytes)
    img = cv2.imdecode(
        np.frombuffer(frame_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if img is None:
        return

    orig_h, orig_w = img.shape[:2]

    # ── SCRFD detection (gRPC binary, no base64) ───────────────────────────────
    raw_boxes = run_detection(frame_bytes)

    if raw_boxes.shape[0] == 0:
        return

    FACES_DETECTED.labels(camera_id=camera_id).inc(raw_boxes.shape[0])

    tracker      = get_tracker(camera_id)
    events       = []
    total_faces  = 0
    skipped      = 0

    if tracker is not None:
        # ── ByteTrack path ─────────────────────────────────────────────────────
        tracked_objs = tracker.update(raw_boxes, img_size=(orig_h, orig_w))

        for obj in tracked_objs:
            total_faces += 1
            track_id = int(obj.track_id)
            x1, y1, x2, y2 = [int(v) for v in obj.tlbr]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            frame_count    = get_and_increment_frame_count(camera_id, track_id)
            existing_state = get_track_state(camera_id, track_id)

            need_recognition = (
                existing_state is None
                or (
                    not existing_state.get("matched", False)
                    and frame_count % RECONFIRM_INTERVAL == 0
                )
            )

            if need_recognition:
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size > 0 and (y2 - y1) >= 20 and (x2 - x1) >= 20:
                    RECOGNITION_CALLS.labels(camera_id=camera_id).inc()
                    embedding = run_recognition(face_crop)
                    identity  = search_embedding(embedding, camera_id, track_id)
                    state     = {**identity, "frame_count": frame_count}
                    set_track_state(camera_id, track_id, state)
                else:
                    identity = existing_state or {"person": "unknown", "matched": False}
            else:
                identity = existing_state or {"person": "unknown", "matched": False}
                skipped += 1

            RECOGNITION_RESULTS.labels(
                camera_id=camera_id,
                result="matched" if identity.get("matched") else "unknown",
            ).inc()

            events.append({
                "track_id":   track_id,
                "person":     identity.get("person", "unknown"),
                "confidence": round(float(identity.get("confidence", 0.0)), 4),
                "matched":    identity.get("matched", False),
                "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        ACTIVE_TRACKS.labels(camera_id=camera_id).set(len(tracked_objs))
        if total_faces > 0:
            BYTETRACK_SKIP_RATIO.labels(camera_id=camera_id).set(
                skipped / total_faces
            )

    else:
        # ── Fallback: no ByteTrack — full inference on every detection ─────────
        for i, box in enumerate(raw_boxes):
            x1, y1, x2, y2, score = box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = img[y1:y2, x1:x2]
            if face_crop.size < 400:  # < 20×20
                continue

            RECOGNITION_CALLS.labels(camera_id=camera_id).inc()
            embedding = run_recognition(face_crop)
            identity  = search_embedding(embedding, camera_id)

            events.append({
                "track_id":   i,
                "person":     identity.get("person", "unknown"),
                "confidence": round(float(identity.get("confidence", 0.0)), 4),
                "matched":    identity.get("matched", False),
                "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    # ── Publish recognition events ─────────────────────────────────────────────
    if events:
        try:
            producer.send(
                topic=f"camera.{camera_id}.events",
                value={
                    "camera_id":  camera_id,
                    "frame_seq":  frame_seq,
                    "timestamp":  timestamp,
                    "worker_id":  WORKER_ID,
                    "faces":      events,
                },
                key=camera_id,
            )
        except Exception as exc:
            logger.error("Failed to publish events: %s", exc)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    start_metrics_server()

    logger.info("Inference worker %s starting…", WORKER_ID)

    # Wait for Triton before consuming
    get_triton()
    get_redis()

    consumer = make_consumer()
    producer = make_producer()

    logger.info("Worker ready — consuming camera.*.frames")

    for message in consumer:
        try:
            headers   = {k: v.decode() for k, v in (message.headers or [])}
            camera_id = headers.get("camera_id") or message.topic.split(".")[1]
            frame_seq = int(headers.get("frame_seq", "0"))
            timestamp = headers.get("timestamp_utc", datetime.now(timezone.utc).isoformat())

            process_frame(
                producer    = producer,
                camera_id   = camera_id,
                frame_bytes = message.value,
                frame_seq   = frame_seq,
                timestamp   = timestamp,
            )
        except Exception as exc:
            logger.error("Unhandled error processing message: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
