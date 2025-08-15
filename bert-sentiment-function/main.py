# Final, Definitive main.py for 2nd Gen Pub/Sub Functions

import base64
import json
import os
import requests
import hashlib
from google.cloud import bigquery, error_reporting, monitoring_v3

# Initialize clients
error_client = error_reporting.Client()
bq_client = bigquery.Client()
monitoring_client = monitoring_v3.MetricServiceClient()

# Environment variables from deployment
BERT_API_URL = os.environ.get('BERT_API_URL')
GCP_PROJECT = os.environ.get('GCP_PROJECT')
DATASET_NAME = "bert_predictions"
TABLE_NAME = "prediction_history"

def bert_sentiment_processor(event, context):
    """
    Triggered by a Pub/Sub message to process a review for sentiment analysis.
    This version is definitively corrected for the observed 2nd Gen Cloud Function event structure.
    """
    try:
        # THIS IS THE FINAL FIX: 
        # The logs show the message data is directly in event['data']
        if 'data' in event:
            pubsub_message_data = base64.b64decode(event['data']).decode('utf-8')
            message_data = json.loads(pubsub_message_data)
            print(f"Successfully decoded message: {message_data}")
        else:
            print(f"Error: 'data' key not found in event: {event}")
            raise ValueError("Invalid CloudEvent format: 'data' key is missing.")

        review_text = message_data.get('review_text')
        request_id = message_data.get('request_id')
        timestamp = message_data.get('timestamp')

        if not all([review_text, request_id, timestamp]):
            raise ValueError("Missing one or more required fields (review_text, request_id, timestamp) in the message.")

        # Call the BERT API
        print(f"Calling BERT API for request_id: {request_id}")
        api_payload = {"review_text": review_text}
        response = requests.post(BERT_API_URL, json=api_payload, timeout=30)
        response.raise_for_status()
        api_result = response.json()
        print(f"Received API response: {api_result}")

        # Hash the input text to create a unique ID
        text_hash = hashlib.sha256(review_text.encode()).hexdigest()
        
        # Prepare data for BigQuery
        prediction_row = [{
            "timestamp": timestamp,
            "request_id": request_id,
            "review_text": review_text,
            "text_hash": text_hash,
            "predicted_sentiment": api_result.get("sentiment"),
            "confidence": api_result.get("confidence"),
            "processing_time": api_result.get("processing_time_ms")
        }]

        # Insert into BigQuery
        table_id = f"{GCP_PROJECT}.{DATASET_NAME}.{TABLE_NAME}"
        errors = b_client.insert_rows_json(table_id, prediction_row)
        if not errors:
            print(f"Successfully inserted prediction {request_id} into BigQuery.")
        else:
            print(f"BigQuery insert errors: {errors}")

    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred: {e}")
        error_client.report_exception()