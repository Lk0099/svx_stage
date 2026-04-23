#!/usr/bin/env python3
"""
SmartVision-X — Integration Test Suite
=======================================
Tests Stage 1 (pgvector), Stage 2 (streaming), Stage 3 (tracking), Stage 5 (observability).

Usage:
    python scripts/test_integration.py [--base-url http://localhost:8003]
    python scripts/test_integration.py --test stage1  # run only pgvector tests
    python scripts/test_integration.py --test stage5  # run only observability tests

Requirements: requests, Pillow
"""

import argparse
import io
import json
import sys
import time

import requests
from PIL import Image, ImageDraw

BASE_URL      = "http://localhost:8003"
TRITON_URL    = "http://localhost:8000"
PROMETHEUS    = "http://localhost:9090"
GRAFANA       = "http://localhost:3001"
KAFKA_METRICS = "http://localhost:8080"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m→\033[0m"

results = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  {PASS} {name}")
        results["passed"] += 1
    else:
        print(f"  {FAIL} {name}" + (f"  [{detail}]" if detail else ""))
        results["failed"] += 1
    return condition


def make_test_image(width: int = 200, height: int = 200) -> bytes:
    """Create a synthetic test JPEG (solid colour, no real face)."""
    img = Image.new("RGB", (width, height), color=(180, 140, 100))
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 40, 140, 120], fill=(220, 180, 140))  # rough face shape
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ── Stage 1 — pgvector ─────────────────────────────────────────────────────────
def test_stage1():
    print(f"\n{INFO} STAGE 1 — pgvector Identity Store")

    # Health check — postgres must be in checks
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    check("Health endpoint returns 200 or 503", r.status_code in (200, 503))
    body = r.json()
    check("Health response includes 'postgres' check", "postgres" in body.get("checks", {}))
    check("Health reports pgvector store version", "2.0.0" in body.get("version", ""))

    # Register test identity
    img_bytes = make_test_image()
    r = requests.post(
        f"{BASE_URL}/register?name=test_pgvector_01",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=15,
    )
    # May be 400 (no face) or 200 (success) — both acceptable; 500 = failure
    check("Register endpoint responds (not 500)", r.status_code != 500,
          f"Got {r.status_code}: {r.text[:100]}")

    # List identities
    r = requests.get(f"{BASE_URL}/identities", timeout=10)
    check("GET /identities returns 200", r.status_code == 200)
    body = r.json()
    check("Identities response has 'total' and 'identities' keys",
          "total" in body and "identities" in body)

    # Camera registration
    r = requests.post(
        f"{BASE_URL}/cameras",
        json={
            "camera_id":    "test-cam-01",
            "name":         "Test Camera",
            "rtsp_url":     "rtsp://192.168.1.1:554/stream",
            "location":     "Entrance",
            "frame_sample": 3,
        },
        timeout=10,
    )
    check("POST /cameras returns 200", r.status_code == 200)

    r = requests.get(f"{BASE_URL}/cameras", timeout=10)
    check("GET /cameras returns 200", r.status_code == 200)

    # Internal search endpoint
    r = requests.post(
        f"{BASE_URL}/internal/search-embedding",
        json={"embedding": [0.0] * 512, "threshold": 0.45},
        timeout=10,
    )
    check("POST /internal/search-embedding returns 200", r.status_code == 200)


# ── Stage 5 — Observability ────────────────────────────────────────────────────
def test_stage5():
    print(f"\n{INFO} STAGE 5 — Observability Stack")

    # Prometheus metrics endpoint on face-service
    r = requests.get(f"{BASE_URL}/metrics", timeout=10)
    check("GET /metrics returns 200", r.status_code == 200)
    metrics_text = r.text
    check("svx_registered_identities_total metric present",
          "svx_registered_identities_total" in metrics_text)
    check("svx_recognitions_total metric present",
          "svx_recognitions_total" in metrics_text)
    check("svx_recognition_latency_seconds metric present",
          "svx_recognition_latency_seconds" in metrics_text)
    check("svx_db_search_latency_seconds metric present",
          "svx_db_search_latency_seconds" in metrics_text)

    # Triton Prometheus metrics
    r = requests.get(f"{TRITON_URL.replace('8000', '8002')}/metrics", timeout=5)
    check("Triton metrics endpoint reachable",
          r.status_code == 200, f"Got {r.status_code}")
    if r.status_code == 200:
        check("nv_gpu_utilization metric in Triton output",
              "nv_gpu_utilization" in r.text)

    # Prometheus server
    r = requests.get(f"{PROMETHEUS}/-/ready", timeout=5)
    check("Prometheus server ready", r.status_code == 200,
          "Is prometheus container running?")

    # Grafana
    r = requests.get(f"{GRAFANA}/api/health", timeout=5)
    check("Grafana health OK", r.status_code == 200,
          "Is grafana container running?")


# ── Stage 2 — Streaming ────────────────────────────────────────────────────────
def test_stage2():
    print(f"\n{INFO} STAGE 2 — Camera Stream Ingestion")

    # Check Kafka is reachable (via face-service health check)
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    body = r.json()
    checks = body.get("checks", {})
    # Kafka health is not directly in face-service checks, but Prometheus lag exporter
    # confirms Kafka connectivity

    # Verify ingest-service metrics endpoint (if cam-01 running)
    try:
        r = requests.get("http://localhost:9091/metrics", timeout=3)
        check("Ingest worker metrics reachable (cam-01)", r.status_code == 200)
        check("svx_ingest_frames_published_total in ingest metrics",
              "svx_ingest_frames_published_total" in r.text)
    except requests.exceptions.ConnectionError:
        print(f"  {INFO} Ingest metrics not reachable (cam-01 may not be configured)")


# ── Stage 3 — Tracking ─────────────────────────────────────────────────────────
def test_stage3():
    print(f"\n{INFO} STAGE 3 — Inference Worker + ByteTrack")

    # Check inference-worker metrics endpoint
    try:
        r = requests.get("http://localhost:9092/metrics", timeout=3)
        check("Inference worker metrics reachable", r.status_code == 200)
        check("svx_frames_processed_total in worker metrics",
              "svx_frames_processed_total" in r.text)
        check("svx_bytetrack_skip_ratio in worker metrics",
              "svx_bytetrack_skip_ratio" in r.text)
    except requests.exceptions.ConnectionError:
        print(f"  {INFO} Inference worker metrics not reachable (check container)")


# ── Triton model readiness ─────────────────────────────────────────────────────
def test_triton():
    print(f"\n{INFO} Triton Model Readiness")

    for model in ("face_detection", "face_recognition"):
        try:
            r = requests.get(
                f"{TRITON_URL}/v2/models/{model}/ready",
                timeout=5,
            )
            check(f"Triton model '{model}' is ready", r.status_code == 200)
        except requests.exceptions.ConnectionError:
            check(f"Triton model '{model}' reachable", False, "Connection refused")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--test", choices=["all", "stage1", "stage2", "stage3", "stage5", "triton"],
                        default="all")
    args = parser.parse_args()
    BASE_URL = args.base_url

    print("=" * 60)
    print("SmartVision-X Integration Tests")
    print("=" * 60)

    tests = {
        "triton": test_triton,
        "stage1": test_stage1,
        "stage5": test_stage5,
        "stage2": test_stage2,
        "stage3": test_stage3,
    }

    if args.test == "all":
        for name, fn in tests.items():
            fn()
    else:
        tests[args.test]()

    print("\n" + "=" * 60)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")
    print("=" * 60)
    sys.exit(0 if results["failed"] == 0 else 1)
