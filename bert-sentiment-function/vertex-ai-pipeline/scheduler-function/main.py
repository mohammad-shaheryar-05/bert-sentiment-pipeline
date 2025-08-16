import os
import json
import statistics
from datetime import datetime
from google.cloud import bigquery
from google.cloud import aiplatform

# --- Global Clients and Constants ---
# It's safe to initialize clients and define constants here.
PROJECT_ID = os.environ.get('GCP_PROJECT', 'ms-gcu-dissertation')
REGION = "us-central1"
BUCKET_NAME = "ms-gcu-dissertation-bert-predictions"
DRIFT_THRESHOLD = 0.15
bigquery_client = bigquery.Client()

def check_drift_and_retrain(event, context):
    """
    A Cloud Function triggered by Cloud Scheduler to check for model drift
    and trigger a Vertex AI retraining pipeline if needed.
    """
    print("Starting daily model drift check...")
    try:
        # 1. Query recent predictions
        query = f"""
        SELECT confidence
        FROM `{PROJECT_ID}.bert_predictions.prediction_history`
        WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 7 DAY)
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        results = bigquery_client.query(query).result()

        if results.total_rows < 100:
            print(f"Not enough data for drift detection ({results.total_rows} predictions). Exiting.")
            return "SUCCESS: Not enough data.", 200

        # 2. Calculate average confidence
        confidences = [row.confidence for row in results]
        avg_confidence = statistics.mean(confidences)
        print(f"Average confidence of last {len(confidences)} predictions: {avg_confidence:.3f}")

        # 3. Detect drift
        baseline_confidence = 0.8
        drift_detected = avg_confidence < (baseline_confidence - DRIFT_THRESHOLD)

        if drift_detected:
            print(f"Drift DETECTED! Average confidence ({avg_confidence:.3f}) is below threshold.")
            # FIX: Call the trigger function only when needed
            trigger_retraining_pipeline()
        else:
            print(f"No drift detected. Average confidence ({avg_confidence:.3f}) is acceptable.")
        
        return "SUCCESS: Drift check completed.", 200

    except Exception as e:
        print(f"ERROR: Drift detection failed: {e}")
        return "ERROR: Drift check failed.", 500

def trigger_retraining_pipeline():
    """Triggers the Vertex AI retraining pipeline."""
    print("Triggering Vertex AI retraining pipeline...")
    
    # FIX: All executable code is now safely inside the function.
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # The datetime.now() call is now executed at runtime, not import time.
    job_id = f"bert-retraining-drift-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    pipeline = aiplatform.PipelineJob(
        display_name="bert-retraining-triggered-by-drift",
        template_path=f"gs://{BUCKET_NAME}/pipelines/bert_retraining_pipeline.yaml",
        job_id=job_id,
        parameter_values={
            "project_id": PROJECT_ID,
            "dataset_name": "bert_predictions",
            "base_model_path": "bert-base-uncased",
            "region": REGION
        },
        enable_caching=False
    )
    
    pipeline.submit()
    print(f"Successfully submitted pipeline job with ID: {job_id}")