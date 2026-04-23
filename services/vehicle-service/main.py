from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import redis
import json
from kafka import KafkaProducer
import time

app = FastAPI()

# Redis
r = redis.Redis(host="redis", port=6379, decode_responses=True)

producer = None

def get_kafka_producer():
    global producer
    if producer is None:
        while True:
            try:
                producer = KafkaProducer(
                    bootstrap_servers="kafka:9092",
                    value_serializer=lambda v: json.dumps(v).encode("utf-8")
                )
                print("✅ Connected to Kafka")
                break
            except Exception as e:
                print("⏳ Waiting for Kafka...", e)
                time.sleep(5)
    return producer

class VehicleData(BaseModel):
    vehicle_id: str
    lat: float
    lon: float
    speed: float

@app.post("/vehicle/location")
async def update_location(data: VehicleData):
    payload = {
        "vehicle_id": data.vehicle_id,
        "lat": data.lat,
        "lon": data.lon,
        "speed": data.speed,
        "timestamp": str(datetime.utcnow())
    }

    # Store in Redis
    r.set(f"vehicle:{data.vehicle_id}", json.dumps(payload))

    # Send to Kafka (lazy init)
    kafka = get_kafka_producer()
    kafka.send("vehicle_location", payload)

    return {"status": "updated", "data": payload}
