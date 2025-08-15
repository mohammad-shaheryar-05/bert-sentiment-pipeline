import base64
import json
from google.cloud import storage
from datetime import datetime

def pubsub_to_gcs(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic."""
    # Decode the Pub/Sub message
    message = base64.b64decode(event['data']).decode('utf-8')
    data = json.loads(message)

    # Prepare filename
    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S%f')
    filename = f'review-{timestamp}.json'

    # Save to Cloud Storage
    client = storage.Client()
    bucket = client.bucket('sentiment-data-bucket')  # Replace with your actual bucket name
    blob = bucket.blob(f'reviews/{filename}')
    blob.upload_from_string(json.dumps(data), content_type='application/json')

    print(f'Uploaded: {filename}')
