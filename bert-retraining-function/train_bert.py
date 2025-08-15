"""
Vertex AI Pipeline for Automated BERT Model Retraining
"""

import json
import base64
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import run_v2
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT', 'ms-gcu-dissertation')
REGION = 'us-central1'
BUCKET_NAME = 'ms-gcu-dissertation-bert-predictions'
PIPELINE_NAME = 'bert-retraining-pipeline'
CLOUD_RUN_SERVICE = 'bert-sentiment-service'

# Initialize clients
storage_client = storage.Client()
aiplatform.init(project=PROJECT_ID, location=REGION)

def drift_alert_handler(cloud_event):
    """
    Cloud Function triggered by drift alert messages.
    Initiates the retraining pipeline.
    """
    try:
        # Decode the Pub/Sub message
        pubsub_message = base64.b64decode(cloud_event.data['message']['data']).decode('utf-8')
        alert_data = json.loads(pubsub_message)
        
        logger.info(f"Received drift alert: {alert_data}")
        
        if alert_data.get('alert_type') == 'model_drift':
            drift_score = alert_data.get('drift_score', 0)
            
            # Start retraining pipeline
            job_id = start_retraining_pipeline(alert_data)
            
            logger.info(f"Started retraining pipeline with job ID: {job_id}")
            
    except Exception as e:
        logger.error(f"Error handling drift alert: {str(e)}")

def start_retraining_pipeline(alert_data):
    """
    Start the Vertex AI training pipeline.
    """
    try:
        # Prepare training data
        training_data_uri = prepare_training_data()
        
        # Create custom training job
        job = aiplatform.CustomJob.from_local_script(
            display_name=f"bert-retraining-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            script_path="train_bert.py",
            container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
            requirements=["transformers==4.30.0", "torch==2.0.1", "scikit-learn", "pandas"],
            model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-13:latest",
            args=[
                f"--training-data-uri={training_data_uri}",
                f"--output-model-uri=gs://{BUCKET_NAME}/models/retrained-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                f"--drift-score={alert_data.get('drift_score', 0)}",
                "--epochs=3",
                "--batch-size=16",
                "--learning-rate=2e-5"
            ],
            replica_count=1,
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1
        )
        
        # Submit the job
        job.submit()
        
        # Monitor training job
        job.wait_for_resource_creation()
        
        return job.resource_name
        
    except Exception as e:
        logger.error(f"Failed to start retraining pipeline: {str(e)}")
        raise

def prepare_training_data():
    """
    Prepare training data from recent predictions for retraining.
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Collect recent predictions for training
        training_data = []
        
        # Get predictions from the last 30 days
        blobs = bucket.list_blobs(prefix="predictions/")
        
        for blob in blobs:
            if blob.name.endswith('.json') and 'error_' not in blob.name:
                try:
                    content = json.loads(blob.download_as_text())
                    if 'prediction' in content:
                        # Extract text and predicted label for training
                        training_data.append({
                            "text": content["input"]["review_text"],
                            "label": content["prediction"]["predicted_sentiment"],
                            "confidence": content["prediction"].get("confidence", 0.5)
                        })
                except:
                    continue
        
        # Save training data to GCS
        training_data_uri = f"gs://{BUCKET_NAME}/training_data/retraining_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        # Convert to JSONL format
        jsonl_data = "\n".join([json.dumps(item) for item in training_data])
        
        blob = bucket.blob(training_data_uri.replace(f"gs://{BUCKET_NAME}/", ""))
        blob.upload_from_string(jsonl_data, content_type='application/jsonl')
        
        logger.info(f"Prepared training data: {len(training_data)} samples at {training_data_uri}")
        
        return training_data_uri
        
    except Exception as e:
        logger.error(f"Failed to prepare training data: {str(e)}")
        raise

def model_deployment_handler(cloud_event):
    """
    Handle model deployment after successful training.
    """
    try:
        # This would be triggered by Vertex AI training completion
        pubsub_message = base64.b64decode(cloud_event.data['message']['data']).decode('utf-8')
        deployment_data = json.loads(pubsub_message)
        
        if deployment_data.get('status') == 'training_completed':
            model_uri = deployment_data.get('model_uri')
            
            # Deploy new model to Cloud Run
            deploy_model_to_cloud_run(model_uri)
            
    except Exception as e:
        logger.error(f"Error in model deployment handler: {str(e)}")

def deploy_model_to_cloud_run(model_uri):
    """
    Deploy the newly trained model to Cloud Run.
    """
    try:
        # Create new Docker image with the retrained model
        new_image_uri = build_new_docker_image(model_uri)
        
        # Update Cloud Run service with new image
        client = run_v2.ServicesClient()
        
        service_name = f"projects/{PROJECT_ID}/locations/{REGION}/services/{CLOUD_RUN_SERVICE}"
        
        # Get current service
        current_service = client.get_service(name=service_name)
        
        # Update with new image
        current_service.spec.template.spec.template.spec.containers[0].image = new_image_uri
        
        # Deploy with traffic split (90% old, 10% new for testing)
        current_service.spec.traffic = [
            {"percent": 90, "revision": current_service.status.latest_ready_revision_name},
            {"percent": 10, "type_": run_v2.TrafficTargetAllocationType.TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST}
        ]
        
        # Update the service
        operation = client.update_service(service=current_service)
        operation.result()  # Wait for completion
        
        logger.info(f"Deployed new model to Cloud Run with 10% traffic split")
        
        # Schedule validation checks
        schedule_model_validation()
        
    except Exception as e:
        logger.error(f"Failed to deploy model to Cloud Run: {str(e)}")
        raise

def build_new_docker_image(model_uri):
    """
    Build new Docker image with retrained model.
    """
    # This would trigger Cloud Build to create a new image
    # with the retrained model files
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    new_image_uri = f"gcr.io/{PROJECT_ID}/bert-sentiment:retrained-{timestamp}"
    
    # Cloud Build configuration would be triggered here
    # For now, return the expected image URI
    return new_image_uri

def schedule_model_validation():
    """
    Schedule validation tests for the new model deployment.
    """
    try:
        # This would schedule validation jobs to test the new model
        # against a holdout dataset and compare performance
        
        validation_job = {
            "job_type": "model_validation",
            "timestamp": datetime.now().isoformat(),
            "model_version": "retrained",
            "validation_tests": [
                "accuracy_test",
                "latency_test", 
                "drift_validation"
            ]
        }
        
        # Save validation job to storage
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"validation_jobs/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        blob.upload_from_string(json.dumps(validation_job, indent=2))
        
        logger.info("Scheduled model validation tests")
        
    except Exception as e:
        logger.error(f"Failed to schedule model validation: {str(e)}")

# Training script that would be used by Vertex AI
TRAINING_SCRIPT = '''
"""
train_bert.py - BERT model retraining script for Vertex AI
"""

import argparse
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from google.cloud import storage
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_training_data(data_uri):
    """Load training data from GCS"""
    storage_client = storage.Client()
    bucket_name = data_uri.split('/')[2]
    blob_path = '/'.join(data_uri.split('/')[3:])
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    data = []
    for line in blob.download_as_text().strip().split('\n'):
        data.append(json.loads(line))
    
    return data

def prepare_datasets(data, tokenizer):
    """Prepare training and validation datasets"""
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    texts = [item['text'] for item in data]
    labels = [label_map[item['label']] for item in data if item['label'] in label_map]
    
    # Filter texts to match labels (in case some labels were invalid)
    valid_indices = [i for i, item in enumerate(data) if item['label'] in label_map]
    texts = [texts[i] for i in valid_indices]
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data-uri', required=True)
    parser.add_argument('--output-model-uri', required=True)
    parser.add_argument('--drift-score', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    
    args = parser.parse_args()
    
    logger.info(f"Starting retraining with drift score: {args.drift_score}")
    
    # Load data
    logger.info(f"Loading training data from {args.training_data_uri}")
    training_data = load_training_data(args.training_data_uri)
    
    # Initialize tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(training_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save the model
    logger.info(f"Saving model to {args.output_model_uri}")
    
    # Save locally first
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')
    
    # Upload to GCS
    storage_client = storage.Client()
    bucket_name = args.output_model_uri.split('/')[2]
    model_path = '/'.join(args.output_model_uri.split('/')[3:])
    
    bucket = storage_client.bucket(bucket_name)
    
    # Upload all model files
    for file_name in os.listdir('./trained_model'):
        blob = bucket.blob(f"{model_path}/{file_name}")
        blob.upload_from_filename(f"./trained_model/{file_name}")
    
    # Save training metadata
    metadata = {
        "training_completed": True,
        "drift_score": args.drift_score,
        "eval_accuracy": eval_results['eval_accuracy'],
        "model_uri": args.output_model_uri,
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate
    }
    
    metadata_blob = bucket.blob(f"{model_path}/training_metadata.json")
    metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
'''