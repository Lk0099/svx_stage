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
                enable_auto_commit=True,
                group_id="tracking-group"
            )
            print("🚀 Tracking Engine Connected to Kafka")
            return consumer
        except Exception as e:
            print("⏳ Waiting for Kafka (tracking)...", e)
            time.sleep(5)

consumer = create_consumer()

print("🚀 Tracking Engine Started...")

for message in consumer:
    data = message.value
    vehicle_id = data["vehicle_id"]

    key = f"tracking:{vehicle_id}"
    r.rpush(key, json.dumps(data))

    print(f"📍 Tracking update: {data}", flush=True)
