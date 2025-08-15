"""
Vertex AI Pipeline for BERT Model Retraining
This pipeline automatically retrains the BERT sentiment model when drift is detected.
"""

from kfp import dsl, compiler
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple

from google.cloud import aiplatform
import os

# THIS COMPONENT IS NOW CORRECTED
@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery>=3.11.0", 
        "google-cloud-storage>=2.10.0", 
        "transformers>=4.30.0", 
        "torch>=2.0.0", 
        "scikit-learn>=1.3.0", 
        "pandas>=2.0.0",
        "db-dtypes"  # <-- THIS IS THE REQUIRED FIX
    ]
)
def extract_training_data(
    project_id: str,
    dataset_name: str,
    days_back: int,
    output_data: Output[Dataset]
) -> dict:
    """Extract recent prediction data for retraining."""
    import pandas as pd
    from google.cloud import bigquery
    import json
    
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT review_text, predicted_sentiment, confidence, timestamp
    FROM `{project_id}.{dataset_name}.prediction_history`
    WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL {days_back} DAY) AND confidence < 0.7
    ORDER BY timestamp DESC LIMIT 10000
    """
    df = client.query(query).to_dataframe()
    df.to_csv(output_data.path, index=False)
    stats = {
        "total_samples": len(df),
        "sentiment_distribution": df['predicted_sentiment'].value_counts().to_dict(),
        "avg_confidence": float(df['confidence'].mean())
    }
    return stats

@component(
    base_image="python:3.10",
    packages_to_install=[
        "transformers>=4.30.0", "torch>=2.0.0", "scikit-learn>=1.3.0",
        "pandas>=2.0.0", "datasets>=2.14.0"
    ]
)
def prepare_training_data(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset]
) -> dict:
    """Prepare and split data for retraining."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(input_data.path)
    label_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
    df['label'] = df['predicted_sentiment'].map(label_map)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df.to_csv(train_data.path, index=False)
    val_df.to_csv(val_data.path, index=False)
    return {"train_samples": len(train_df), "val_samples": len(val_df), "classes": list(label_map.keys())}

@component(
    base_image="python:3.10",
    packages_to_install=[
        "transformers>=4.30.0", "torch>=2.0.0", "scikit-learn>=1.3.0",
        "pandas>=2.0.0", "datasets>=2.14.0"
    ]
)
def retrain_bert_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    base_model_path: str,
    model_output: Output[Model],
    metrics_output: Output[Metrics]
) -> dict:
    """Retrain BERT model with new data."""
    import pandas as pd
    import torch
    from transformers import (BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding)
    from torch.utils.data import Dataset as TorchDataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import json
    
    class SentimentDataset(TorchDataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            text = str(self.texts.iloc[idx])
            encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)}

    train_df, val_df = pd.read_csv(train_data.path), pd.read_csv(val_data.path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    train_dataset, val_dataset = SentimentDataset(train_df['review_text'], train_df['label'], tokenizer), SentimentDataset(val_df['review_text'], val_df['label'], tokenizer)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred; predictions = predictions.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        return {'accuracy': accuracy_score(labels, predictions), 'f1': f1, 'precision': precision, 'recall': recall}

    trainer = Trainer(model=model, args=TrainingArguments(output_dir='/tmp/retrained_model', num_train_epochs=2, per_device_train_batch_size=16, logging_steps=100, evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="eval_accuracy"), train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    training_result = trainer.train()
    eval_result = trainer.evaluate()
    model.save_pretrained(model_output.path)
    tokenizer.save_pretrained(model_output.path)
    metrics = {"eval_accuracy": float(eval_result["eval_accuracy"]), "eval_f1": float(eval_result["eval_f1"])}
    with open(f"{metrics_output.path}/metrics.json", "w") as f: json.dump(metrics, f)
    return metrics

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-storage>=2.10.0", "pandas>=2.0.0"]
)
def deploy_model_to_cloud_run(
    model: Input[Model],
    bucket_name: str
) -> NamedTuple("Outputs", [("model_uploaded", str), ("deployment_status", str)]):
    """Deploy retrained model to Cloud Run."""
    from google.cloud import storage
    import zipfile, tempfile, os, pandas as pd
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip, 'w') as zipf:
            for root, _, files in os.walk(model.path):
                for file in files:
                    file_path, arcname = os.path.join(root, file), os.path.relpath(file_path, model.path)
                    zipf.write(file_path, arcname)
        temp_zip_path = temp_zip.name
    
    blob = bucket.blob(f"models/retrained_bert_{int(pd.Timestamp.now().timestamp())}.zip")
    blob.upload_from_filename(temp_zip_path)
    os.unlink(temp_zip_path)
    
    model_gcs_path = f"gs://{bucket_name}/{blob.name}"
    deployment_status = "model_uploaded_ready_for_deployment"
    
    return (model_gcs_path, deployment_status)

@pipeline(
    name="bert-sentiment-retraining-pipeline",
    description="Automated BERT sentiment model retraining pipeline"
)
def bert_retraining_pipeline(
    project_id: str = "ms-gcu-dissertation",
    dataset_name: str = "bert_predictions",
    days_back: int = 30,
    base_model_path: str = "bert-base-uncased",
    bucket_name: str = "ms-gcu-dissertation-bert-predictions"
) -> str:
    """Complete BERT retraining pipeline."""
    
    extract_task = extract_training_data(project_id=project_id, dataset_name=dataset_name, days_back=days_back)
    prepare_task = prepare_training_data(input_data=extract_task.outputs["output_data"])
    
    retrain_task = retrain_bert_model(
        train_data=prepare_task.outputs["train_data"], 
        val_data=prepare_task.outputs["val_data"], 
        base_model_path=base_model_path
    )
    
    deploy_task = deploy_model_to_cloud_run(
        model=retrain_task.outputs["model_output"], 
        bucket_name=bucket_name
    )
    
    return deploy_task.outputs["model_uploaded"]

def compile_pipeline():
    """Compile the pipeline to YAML."""
    compiler.Compiler().compile(
        pipeline_func=bert_retraining_pipeline,
        package_path="bert_retraining_pipeline.yaml"
    )