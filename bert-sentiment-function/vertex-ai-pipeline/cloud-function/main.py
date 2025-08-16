import json
import base64
import requests
import logging
from datetime import datetime
from google.cloud import storage
from google.cloud import error_reporting
from google.cloud import monitoring_v3
from google.cloud import bigquery
import traceback
import os
import statistics
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
storage_client = storage.Client()
error_client = error_reporting.Client()
monitoring_client = monitoring_v3.MetricServiceClient()
bigquery_client = bigquery.Client()

# Configuration
BERT_API_URL = os.environ.get('BERT_API_URL', "https://bert-sentiment-service-121496194098.us-central1.run.app")
BUCKET_NAME = os.environ.get('BUCKET_NAME', "ms-gcu-dissertation-bert-predictions")
PROJECT_ID = os.environ.get('GCP_PROJECT', 'ms-gcu-dissertation')
PREDICTIONS_FOLDER = "predictions"
DRIFT_THRESHOLD = float(os.environ.get('DRIFT_THRESHOLD', '0.15'))

def bert_sentiment_processor(cloud_event, context):
    """
    Enhanced Cloud Function with monitoring and drift detection.
    """
    start_time = datetime.now()
    message_data = {}

    try:
        # FINAL, CORRECT FIX: Based on the logs, the base64 message is directly in the 'data' key.
        if 'data' not in cloud_event:
            raise ValueError(f"Invalid Pub/Sub message format: 'data' key is missing. Event: {cloud_event}")
        
        pubsub_message = base64.b64decode(cloud_event['data']).decode('utf-8')
        message_data = json.loads(pubsub_message)
        
        logger.info(f"Processing message: {message_data}")

        review_text = message_data.get('review_text', '')
        request_id = message_data.get('request_id', f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        timestamp = message_data.get('timestamp', datetime.now().isoformat())

        if not review_text.strip():
            raise ValueError("Empty review_text provided")

        prediction_result = call_bert_api(review_text)
        processing_time = (datetime.now() - start_time).total_seconds()

        result = {
            "request_id": request_id, "timestamp": timestamp,
            "input": {"review_text": review_text, "text_hash": hashlib.md5(review_text.encode()).hexdigest()[:8]},
            "prediction": prediction_result,
            "metadata": {
                "api_url": BERT_API_URL, "function_name": os.environ.get('K_SERVICE', 'bert-sentiment-processor'),
                "message_id": context.event_id, "processing_time_seconds": processing_time
            }
        }

        save_to_storage(result, request_id)
        store_prediction_bigquery(result)

        logger.info(f"Successfully processed request {request_id}")

    except Exception as e:
        error_message = f"Error processing Pub/Sub message: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        error_client.report_exception()
        try:
            error_result = {
                "request_id": message_data.get('request_id', 'unknown'), "timestamp": datetime.now().isoformat(),
                "status": "error", "error": str(e), "input": message_data or "Unable to parse message",
                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
            }
            save_to_storage(error_result, f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        except:
            logger.error("Failed to save error information to storage")

def call_bert_api(review_text):
    """Call the BERT sentiment analysis API."""
    payload = {"review_text": review_text}
    response = requests.post(
        f"{BERT_API_URL}/predict",
        json=payload, headers={"Content-Type": "application/json"}, timeout=60
    )
    response.raise_for_status()
    return response.json()

def save_to_storage(data, request_id):
    """Save prediction results to Cloud Storage."""
    bucket = storage_client.bucket(BUCKET_NAME)
    timestamp = datetime.now().strftime('%Y/%m/%d')
    filename = f"{PREDICTIONS_FOLDER}/{timestamp}/{request_id}.json"
    blob = bucket.blob(filename)
    blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
    logger.info(f"Saved prediction to gs://{BUCKET_NAME}/{filename}")

def store_prediction_bigquery(result):
    """Store prediction in BigQuery for drift analysis."""
    dataset_id, table_id = "bert_predictions", "prediction_history"
    row = {
        "timestamp": datetime.now().isoformat(), "request_id": result['request_id'],
        "review_text": result['input']['review_text'], "text_hash": result['input']['text_hash'],
        "predicted_sentiment": result['prediction']['predicted_sentiment'],
        "confidence": result['prediction'].get('confidence', 0),
        "processing_time": result['metadata']['processing_time_seconds']
    }
    table_ref = bigquery_client.dataset(dataset_id).table(table_id)
    errors = bigquery_client.insert_rows_json(table_ref, [row])
    if errors:
        logger.error(f"BigQuery insert errors: {errors}")
    else:
        logger.info("Stored prediction in BigQuery")