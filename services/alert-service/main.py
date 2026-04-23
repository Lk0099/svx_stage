"""
Alert Service — Status API
==========================
Provides a health and alert-queue status endpoint.
The alert processing logic runs in consumer.py (launched by Dockerfile CMD).
"""

from fastapi import FastAPI
import redis
import json

app = FastAPI(title="SmartVision-X Alert Service", version="1.0.0")

r = redis.Redis(host="redis", port=6379, decode_responses=True)


@app.get("/health")
def health():
    try:
        r.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "unreachable"

    return {"service": "alert-service", "redis": redis_status}


@app.get("/alerts")
def get_alerts(limit: int = 50):
    """Return the latest alerts from the Redis queue (newest first)."""
    try:
        raw = r.lrange("alerts", -limit, -1)
        alerts = [json.loads(item) for item in reversed(raw)]
    except Exception as exc:
        return {"error": str(exc), "alerts": []}
    return {"total": len(alerts), "alerts": alerts}


@app.delete("/alerts")
def clear_alerts():
    """Clear all alerts from the queue."""
    r.delete("alerts")
    return {"status": "cleared"}
