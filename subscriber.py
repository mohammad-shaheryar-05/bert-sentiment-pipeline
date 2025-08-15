# subscriber.py
from google.cloud import pubsub_v1, storage
import json
import datetime

PROJECT_ID = "ms-gcu-dissertation"  
SUBSCRIPTION_ID = "amazon-reviews-sub"
BUCKET_NAME = "amazon-review-data"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
storage_client = storage.Client()

def callback(message):
    print(f"Received message: {message.data}")
    data = json.loads(message.data.decode("utf-8"))
    
    # Create a file name with timestamp
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
    blob_name = f"streamed_reviews/review_{timestamp}.json"
    
    # Upload to Cloud Storage
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")
    
    print(f"Saved to {blob_name} in {BUCKET_NAME}")
    message.ack()

# Start listening
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
print(f"Listening for messages on {subscription_path}...\n")

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
