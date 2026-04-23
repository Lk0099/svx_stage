"""
SmartVision-X — Camera Ingest Service  (Stage 2)
=================================================
Opens RTSP/RTMP camera streams and publishes raw JPEG frames to Kafka.

One container instance handles ONE camera.
Deploy N replicas (one per camera) via Docker Compose scale.

Environment variables:
    CAMERA_ID       Unique camera identifier (e.g. "cam-entrance-01")
    RTSP_URL        Full RTSP stream URL
    FRAME_SAMPLE    Process every Nth frame. Default: 3 (30fps → 10fps)
    JPEG_QUALITY    JPEG encode quality 1–95. Default: 85
    KAFKA_BROKER    Kafka broker address. Default: kafka:9092
    METRICS_PORT    Prometheus metrics HTTP port. Default: 9090

Output Kafka topic:
    camera.{CAMERA_ID}.frames
    Value:   raw JPEG bytes
    Headers: camera_id, frame_seq, timestamp_utc, width, height, fps
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
from kafka import KafkaProducer
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("ingest")

# ── Config ─────────────────────────────────────────────────────────────────────
CAMERA_ID    = os.environ.get("CAMERA_ID")
RTSP_URL     = os.environ.get("RTSP_URL")
FRAME_SAMPLE = int(os.environ.get("FRAME_SAMPLE", "3"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY",  "85"))
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9090"))

if not CAMERA_ID:
    logger.error("CAMERA_ID environment variable is required")
    sys.exit(1)
if not RTSP_URL:
    logger.error("RTSP_URL environment variable is required")
    sys.exit(1)

TOPIC = f"camera.{CAMERA_ID}.frames"

# ── Prometheus metrics ─────────────────────────────────────────────────────────
FRAMES_CAPTURED = Counter(
    "svx_ingest_frames_captured_total",
    "Total frames captured from camera",
    ["camera_id"],
)
FRAMES_PUBLISHED = Counter(
    "svx_ingest_frames_published_total",
    "Total frames published to Kafka",
    ["camera_id"],
)
STREAM_RECONNECTS = Counter(
    "svx_ingest_stream_reconnects_total",
    "Number of RTSP stream reconnections",
    ["camera_id"],
)
STREAM_UP = Gauge(
    "svx_ingest_stream_up",
    "1 if stream is connected, 0 if disconnected",
    ["camera_id"],
)
KAFKA_LAG = Gauge(
    "svx_ingest_kafka_queue_size",
    "Approximate Kafka producer queue depth",
    ["camera_id"],
)
ENCODE_LATENCY = Histogram(
    "svx_ingest_encode_latency_seconds",
    "JPEG encode latency per frame",
    ["camera_id"],
    buckets=[0.001, 0.003, 0.005, 0.01, 0.025, 0.05],
)

LABELS = {"camera_id": CAMERA_ID}


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
        pass  # Suppress HTTP access logs


def start_metrics_server():
    server = HTTPServer(("0.0.0.0", METRICS_PORT), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Metrics server listening on port %d", METRICS_PORT)


# ── Kafka producer ─────────────────────────────────────────────────────────────
def make_producer() -> KafkaProducer:
    """Create Kafka producer with retry loop."""
    while True:
        try:
            p = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=None,          # raw bytes — no serialisation overhead
                key_serializer=str.encode,
                max_request_size=10_485_760,    # 10 MB max frame
                compression_type="lz4",         # fast compression reduces bandwidth ~40%
                acks=1,                         # leader ACK only — frames are lossy-ok
                linger_ms=5,                    # 5ms batch window
                batch_size=65536,
            )
            logger.info("Kafka producer connected to %s", KAFKA_BROKER)
            return p
        except Exception as exc:
            logger.warning("Kafka unavailable (%s) — retry in 5s", exc)
            time.sleep(5)


# ── Stream ─────────────────────────────────────────────────────────────────────
def open_stream(url: str) -> cv2.VideoCapture:
    """Open RTSP stream with retry loop."""
    while True:
        logger.info("Opening stream: %s", url)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # minimal buffer for low latency
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,  5_000)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            logger.info(
                "Stream opened — FPS=%.1f, effective=%.1f (sample=%d)",
                fps, fps / FRAME_SAMPLE, FRAME_SAMPLE,
            )
            STREAM_UP.labels(**LABELS).set(1)
            return cap
        logger.warning("Cannot open stream — retry in 5s")
        cap.release()
        STREAM_UP.labels(**LABELS).set(0)
        STREAM_RECONNECTS.labels(**LABELS).inc()
        time.sleep(5)


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    start_metrics_server()

    logger.info(
        "Ingest worker starting | camera=%s | sample=%d | quality=%d",
        CAMERA_ID, FRAME_SAMPLE, JPEG_QUALITY,
    )

    producer     = make_producer()
    cap          = open_stream(RTSP_URL)
    frame_count  = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        ret, frame = cap.read()

        if not ret:
            logger.warning("[%s] Stream read failed — reconnecting", CAMERA_ID)
            cap.release()
            STREAM_UP.labels(**LABELS).set(0)
            STREAM_RECONNECTS.labels(**LABELS).inc()
            time.sleep(2)
            cap = open_stream(RTSP_URL)
            frame_count = 0
            continue

        frame_count += 1
        FRAMES_CAPTURED.labels(**LABELS).inc()

        # Sample every Nth frame
        if frame_count % FRAME_SAMPLE != 0:
            continue

        # Encode frame to JPEG bytes
        t0 = time.perf_counter()
        ok, encoded = cv2.imencode(".jpg", frame, encode_params)
        ENCODE_LATENCY.labels(**LABELS).observe(time.perf_counter() - t0)

        if not ok:
            logger.warning("[%s] Frame encode failed", CAMERA_ID)
            continue

        frame_bytes = encoded.tobytes()
        h, w = frame.shape[:2]
        now  = datetime.now(timezone.utc).isoformat()

        # Headers carry metadata — value carries only raw image bytes
        headers = [
            ("camera_id",     CAMERA_ID.encode()),
            ("frame_seq",     str(frame_count).encode()),
            ("timestamp_utc", now.encode()),
            ("width",         str(w).encode()),
            ("height",        str(h).encode()),
        ]

        try:
            producer.send(
                topic   = TOPIC,
                value   = frame_bytes,
                key     = CAMERA_ID,
                headers = headers,
            )
            FRAMES_PUBLISHED.labels(**LABELS).inc()

            # Update queue depth gauge periodically
            if frame_count % 100 == 0:
                # kafka-python does not expose queue depth directly;
                # approximate from buffer_memory / batch_size
                pass

        except Exception as exc:
            logger.error("[%s] Kafka publish error: %s", CAMERA_ID, exc)

    cap.release()


if __name__ == "__main__":
    main()
