import os
import base64
import json
import time
import uuid
from google.cloud import storage

import requests

# ENV VARS set at deploy time
CLOUD_RUN_URL = os.environ["https://bert-sentiment-service-121496194098.us-central1.run.app"]  # e.g. https://bert-sentiment-xyz.a.run.app/predict
BUCKET_NAME = os.environ["ms-gcu-dissertation_cloudbuild"]      # e.g. amazon-review-data
PREDICTIONS_PREFIX = os.environ.get("PREDICTIONS_PREFIX", "predictions")
TIMEOUT_SECS = int(os.environ.get("HTTP_TIMEOUT_SECS", "15"))
RETRY_COUNT = int(os.environ.get("HTTP_RETRY_COUNT", "3"))

storage_client = storage.Client()

def _call_cloud_run(review_text: str) -> dict:
    payload = {"review_text": review_text}
    last_exc = None
    for _ in range(RETRY_COUNT):
        try:
            resp = requests.post(CLOUD_RUN_URL, json=payload, timeout=TIMEOUT_SECS)
            if resp.status_code == 200:
                return resp.json()
            else:
                last_exc = RuntimeError(f"Non-200 from API: {resp.status_code} {resp.text}")
        except Exception as e:
            last_exc = e
        time.sleep(1.0)
    raise last_exc

def _save_to_gcs(data: dict):
    # Unique filename per event
    blob_name = f"{PREDICTIONS_PREFIX}/{uuid.uuid4().hex}.json"
    bucket = storage_client.bucket(ms-gcu-dissertation_cloudbuild)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")
    return blob_name

# Entry point for Cloud Functions (Gen 2, Pub/Sub trigger)
def pubsub_to_predict(event, context):
    if "data" not in event:
        print("No data in event; skipping")
        return

    message = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    # Expecting publisher to send { "review_text": "...", ... }
    review_text = message.get("review_text", "")

    if not review_text:
        print("Empty or missing 'review_text'; skipping")
        return

    prediction = _call_cloud_run(review_text)

    # Optionally attach original payload
    output = {
        "input": message,
        "prediction": prediction,
        "api_url": CLOUD_RUN_URL
    }

    blob_path = _save_to_gcs(output)
    print(f"Saved prediction to gs://{ms-gcu-dissertation_cloudbuild}/{blob_path}")
