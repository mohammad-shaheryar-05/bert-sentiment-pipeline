"""
Cloud Composer (Apache Airflow) DAG for BERT Pipeline Management
Handles model training, drift detection, and automated retraining
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator
from airflow.providers.google.cloud.operators.pubsub import PubSubPublishMessageOperator
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator
from airflow.providers.google.cloud.operators.functions import CloudFunctionsInvokeFunctionOperator
from airflow.providers.google.cloud.operators.vertex_ai import (
    CreateCustomPythonPackageTrainingJobOperator,
    CreatePipelineJobOperator
)
from airflow.providers.google.cloud.sensors.bigquery import BigQueryTableExistenceSensor
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
import json

# Configuration
PROJECT_ID = "ms-gcu-dissertation"
REGION = "us-central1"
BUCKET_NAME = "ms-gcu-dissertation-bert-predictions"

# Default DAG arguments
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'bert_pipeline_management',
    default_args=default_args,
    description='BERT Sentiment Analysis Pipeline Management',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'bert', 'sentiment-analysis']
)

# Task 1: Check data quality and volume
check_data_quality = BigQueryCheckOperator(
    task_id='check_data_quality',
    sql=f"""
    SELECT COUNT(*) as prediction_count
    FROM `{PROJECT_ID}.bert_predictions.prediction_history`
    WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 24 HOUR)
    AND prediction_count >= 100  -- Minimum predictions per day
    """,
    dag=dag
)

# Task 2: Calculate drift metrics
def calculate_drift_metrics(**context):
    """Calculate model drift metrics from BigQuery data."""
    from google.cloud import bigquery
    import statistics
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # Query recent predictions
    query = f"""
    SELECT 
        predicted_sentiment,
        confidence,
        timestamp,
        EXTRACT(HOUR FROM timestamp) as hour
    FROM `{PROJECT_ID}.bert_predictions.prediction_history`
    WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 7 DAY)
    ORDER BY timestamp DESC
    LIMIT 5000
    """
    
    results = client.query(query).result()
    data = [dict(row) for row in results]
    
    if len(data) < 100:
        print("Not enough data for drift calculation")
        return {"drift_score": 0, "drift_detected": False}
    
    # Calculate metrics
    confidences = [row['confidence'] for row in data]
    avg_confidence = statistics.mean(confidences)
    confidence_std = statistics.stdev(confidences)
    
    # Sentiment distribution
    sentiment_counts = {}
    for row in data:
        sentiment = row['predicted_sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    total_predictions = len(data)
    sentiment_ratios = {
        sentiment: count / total_predictions 
        for sentiment, count in sentiment_counts.items()
    }
    
    # Simple drift detection (you can make this more sophisticated)
    baseline_confidence = 0.8
    drift_threshold = 0.15
    drift_detected = avg_confidence < (baseline_confidence - drift_threshold)
    
    # Calculate drift score (0-1, higher = more drift)
    drift_score = max(0, (baseline_confidence - avg_confidence) / drift_threshold)
    
    metrics = {
        "avg_confidence": avg_confidence,
        "confidence_std": confidence_std,
        "sentiment_distribution": sentiment_ratios,
        "drift_score": min(1.0, drift_score),
        "drift_detected": drift_detected,
        "total_predictions": total_predictions,
        "calculation_time": datetime.now().isoformat()
    }
    
    # Store metrics in XCom for next tasks
    context['task_instance'].xcom_push(key='drift_metrics', value=metrics)
    
    print(f"Drift metrics calculated: {json.dumps(metrics, indent=2)}")
    return metrics

calculate_drift = PythonOperator(
    task_id='calculate_drift_metrics',
    python_callable=calculate_drift_metrics,
    dag=dag
)

# Task 3: Decide if retraining is needed
def decide_retraining(**context):
    """Decide whether to trigger retraining based on drift metrics."""
    drift_metrics = context['task_instance'].xcom_pull(
        task_ids='calculate_drift_metrics', 
        key='drift_metrics'
    )
    
    if not drift_metrics:
        print("No drift metrics available")
        return 'no_retraining_needed'
    
    drift_detected = drift_metrics.get('drift_detected', False)
    drift_score = drift_metrics.get('drift_score', 0)
    
    print(f"Drift score: {drift_score}, Drift detected: {drift_detected}")
    
    # Check multiple conditions for retraining
    retraining_needed = (
        drift_detected or 
        drift_score > 0.7 or  # High drift score
        drift_metrics.get('avg_confidence', 1.0) < 0.6  # Very low confidence
    )
    
    if retraining_needed:
        print("🔄 Retraining needed - triggering pipeline")
        return 'trigger_retraining'
    else:
        print("✅ Model performance is acceptable - no retraining needed")
        return 'no_retraining_needed'

decide_retraining_branch = BranchPythonOperator(
    task_id='decide_retraining',
    python_callable=decide_retraining,
    dag=dag
)

# Task 4a: No retraining needed
no_retraining_needed = DummyOperator(
    task_id='no_retraining_needed',
    dag=dag
)

# Task 4b: Trigger retraining pipeline
trigger_retraining = CreatePipelineJobOperator(
    task_id='trigger_retraining',
    region=REGION,
    project_id=PROJECT_ID,
    display_name=f"bert-retraining-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    template_path=f"gs://{BUCKET_NAME}/pipelines/bert_retraining_pipeline.yaml",
    job_id=f"bert-retraining-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    parameter_values={
        "project_id": PROJECT_ID,
        "dataset_name": "bert_predictions", 
        "days_back": 30,
        "base_model_path": "bert-base-uncased",
        "service_name": "bert-sentiment-service",
        "region": REGION,
        "bucket_name": BUCKET_NAME
    },
    dag=dag
)

# Task 5: Performance monitoring and alerting
def send_performance_alert(**context):
    """Send performance alerts based on metrics."""
    drift_metrics = context['task_instance'].xcom_pull(
        task_ids='calculate_drift_metrics', 
        key='drift_metrics'
    )
    
    if not drift_metrics:
        return
    
    # Create alert message
    alert_data = {
        "alert_type": "performance_monitoring",
        "timestamp": datetime.now().isoformat(),
        "metrics": drift_metrics,
        "dag_run_id": context['dag_run'].run_id
    }
    
    print(f"Performance alert: {json.dumps(alert_data, indent=2)}")
    
    # You can send this to Slack, email, or other alerting systems
    return alert_data

performance_monitoring = PythonOperator(
    task_id='performance_monitoring',
    python_callable=send_performance_alert,
    trigger_rule='none_failed_or_skipped',  # Run regardless of branch taken
    dag=dag
)

# Task 6: Update model registry and metadata
def update_model_registry(**context):
    """Update model registry with current performance metrics."""
    from google.cloud import storage
    import json
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Get drift metrics
    drift_metrics = context['task_instance'].xcom_pull(
        task_ids='calculate_drift_metrics',
        key='drift_metrics'
    )
    
    # Create registry entry
    registry_entry = {
        "dag_run_id": context['dag_run'].run_id,
        "execution_date": context['execution_date'].isoformat(),
        "performance_metrics": drift_metrics,
        "model_status": "active",
        "next_evaluation": (datetime.now() + timedelta(days=1)).isoformat()
    }
    
    # Save to Cloud Storage
    blob = bucket.blob(f"model_registry/{context['execution_date'].strftime('%Y/%m/%d')}/performance_report.json")
    blob.upload_from_string(json.dumps(registry_entry, indent=2))
    
    # Update latest performance
    latest_blob = bucket.blob("model_registry/latest_performance.json")
    latest_blob.upload_from_string(json.dumps(registry_entry, indent=2))
    
    print(f"Model registry updated: {registry_entry}")
    return registry_entry

update_registry = PythonOperator(
    task_id='update_model_registry',
    python_callable=update_model_registry,
    trigger_rule='none_failed_or_skipped',
    dag=dag
)

# Task 7: Generate performance report
def generate_performance_report(**context):
    """Generate daily performance report."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # Generate daily stats
    query = f"""
    SELECT 
        DATE(timestamp) as date,
        predicted_sentiment,
        COUNT(*) as prediction_count,
        AVG(confidence) as avg_confidence,
        MIN(confidence) as min_confidence,
        MAX(confidence) as max_confidence,
        AVG(processing_time) as avg_processing_time
    FROM `{PROJECT_ID}.bert_predictions.prediction_history`
    WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 24 HOUR)
    GROUP BY date, predicted_sentiment
    ORDER BY date DESC, predicted_sentiment
    """
    
    results = client.query(query).result()
    report_data = [dict(row) for row in results]
    
    # Get drift metrics
    drift_metrics = context['task_instance'].xcom_pull(
        task_ids='calculate_drift_metrics',
        key='drift_metrics'
    )
    
    # Create comprehensive report
    report = {
        "report_date": context['execution_date'].strftime('%Y-%m-%d'),
        "daily_stats": report_data,
        "drift_analysis": drift_metrics,
        "summary": {
            "total_predictions": sum(row['prediction_count'] for row in report_data),
            "avg_confidence_overall": drift_metrics.get('avg_confidence', 0) if drift_metrics else 0,
            "drift_detected": drift_metrics.get('drift_detected', False) if drift_metrics else False
        }
    }
    
    # Store report
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    blob = bucket.blob(f"reports/daily/{context['execution_date'].strftime('%Y/%m/%d')}/performance_report.json")
    blob.upload_from_string(json.dumps(report, indent=2, default=str))
    
    print(f"Performance report generated: {len(report_data)} entries")
    return report

generate_report = PythonOperator(
    task_id='generate_performance_report',
    python_callable=generate_performance_report,
    trigger_rule='none_failed_or_skipped',
    dag=dag
)

# Task dependencies
check_data_quality >> calculate_drift >> decide_retraining_branch

decide_retraining_branch >> [no_retraining_needed, trigger_retraining]

[no_retraining_needed, trigger_retraining] >> performance_monitoring >> update_registry >> generate_report