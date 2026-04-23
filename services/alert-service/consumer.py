from kafka import KafkaConsumer
import json
import redis
import time

r = redis.Redis(host="redis", port=6379, decode_responses=True)

def create_consumer():
    while True:
        try:
            consumer = KafkaConsumer(
                "vehicle_location",
                bootstrap_servers="kafka:9092",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                group_id="None"
            )
            print("🚨 Alert Engine Connected to Kafka")
            return consumer
        except Exception as e:
            print("⏳ Waiting for Kafka (alert)...", e)
            time.sleep(5)

consumer = create_consumer()

print("🚨 Alert Engine Started...")

SPEED_LIMIT = 80

for message in consumer:
    data = message.value
    vehicle_id = data["vehicle_id"]
    speed = data["speed"]

    alerts = []

    if speed > SPEED_LIMIT:
        alerts.append("OVER_SPEED")

    if not (26.8 <= data["lat"] <= 27.0 and 75.6 <= data["lon"] <= 75.9):
        alerts.append("OUT_OF_ZONE")

    if alerts:
        alert_payload = {
            "vehicle_id": vehicle_id,
            "alerts": alerts,
            "data": data
        }

        r.rpush("alerts", json.dumps(alert_payload))

        print(f"🚨 ALERT: {alert_payload}", flush=True)
