"""
Vertex AI Pipeline for BERT Model Retraining (FINAL - All Fixes Included v5)
- Renamed to v5 to break all caching.
- Corrects the TrainingArguments by importing and using the required IntervalStrategy enum.
- Contains all previous fixes for NaN errors and data handling.
"""

from kfp import dsl, compiler
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple
import os

# ---------------------------
# Component: extract_training_data_v5
# ---------------------------
@component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas>=2.0.0", "google-cloud-bigquery>=3.11.0",
        "google-cloud-bigquery-storage", "db-dtypes",
    ],
)
def extract_training_data_v5(
    project_id: str,
    dataset_name: str,
    days_back: int,
    output_data: Output[Dataset],
) -> dict:
    import math
    import pandas as pd
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT review_text, predicted_sentiment, confidence, timestamp
    FROM `{project_id}.{dataset_name}.prediction_history`
    WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL {days_back} DAY)
    ORDER BY timestamp DESC LIMIT 10000
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        df = pd.DataFrame(columns=["review_text", "predicted_sentiment", "confidence", "timestamp"])
        df.to_csv(output_data.path, index=False)
        return {"total_samples": 0, "avg_confidence": 0.0}
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce")
    avg_conf = df["confidence"].mean(skipna=True)
    if math.isnan(avg_conf): avg_conf = 0.0
    df.to_csv(output_data.path, index=False)
    return {"total_samples": len(df), "avg_confidence": float(avg_conf)}

# ---------------------------
# Component: prepare_training_data_v5
# ---------------------------
@component(
    base_image="python:3.10",
    packages_to_install=["pandas>=2.0.0", "scikit-learn>=1.3.0"],
)
def prepare_training_data_v5(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
) -> dict:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(input_data.path)
    if df.empty:
        df.to_csv(train_data.path, index=False)
        df.to_csv(val_data.path, index=False)
        return {"train_samples": 0, "val_samples": 0}
    label_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
    df["label"] = df["predicted_sentiment"].map(label_map)
    df = df.dropna(subset=["review_text", "label"])
    df["label"] = df["label"].astype(int)
    if df.empty or df["label"].nunique() < 2 or df["label"].value_counts().min() < 2:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    else:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_df.to_csv(train_data.path, index=False)
    val_df.to_csv(val_data.path, index=False)
    return {"train_samples": len(train_df), "val_samples": len(val_df)}

# ---------------------------
# Component: retrain_bert_model_v5
# ---------------------------
@component(
    base_image="python:3.10",
    packages_to_install=["pandas>=2.0.0", "torch>=2.0.0", "transformers[torch]==4.30.2", "scikit-learn>=1.3.0", "datasets>=2.14.0", "accelerate>=0.21.0"],
)
def retrain_bert_model_v5(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    base_model_path: str,
    model_output: Output[Model],
    metrics_output: Output[Metrics],
) -> dict:
    import os
    import pandas as pd
    import torch
    # THIS IS THE FINAL FIX: Import IntervalStrategy
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, IntervalStrategy
    from torch.utils.data import Dataset as TorchDataset
    from sklearn.metrics import accuracy_score

    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(val_data.path)
    if train_df.empty or len(train_df) < 10:
        print("Skipping training due to insufficient data.")
        os.makedirs(model_output.path, exist_ok=True)
        with open(os.path.join(model_output.path, "placeholder.txt"), "w") as f: f.write("No training.")
        metrics_output.log_metric("accuracy", 0.0)
        return {"trained": False, "accuracy": 0.0}
    
    class SentimentDataset(TorchDataset):
        def __init__(self, texts, labels, tokenizer): self.texts, self.labels, self.tokenizer = texts, labels, tokenizer
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            item = self.tokenizer(str(self.texts.iloc[idx]), truncation=True, padding="max_length", return_tensors="pt")
            return {"input_ids": item["input_ids"].squeeze(0), "attention_mask": item["attention_mask"].squeeze(0), "labels": int(self.labels.iloc[idx])}

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=3)
    train_dataset = SentimentDataset(train_df["review_text"], train_df["label"], tokenizer)
    val_dataset = SentimentDataset(val_df["review_text"], val_df["label"], tokenizer)

    def compute_metrics(p): return {"accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))}
    
    args = TrainingArguments(
        output_dir="/tmp/model",
        # THIS IS THE FINAL FIX: Use the imported enum, not a string
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        num_train_epochs=1,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()
    eval_metrics = trainer.evaluate()
    model.save_pretrained(model_output.path)
    tokenizer.save_pretrained(model_output.path)
    accuracy = eval_metrics.get("eval_accuracy", 0.0)
    metrics_output.log_metric("accuracy", accuracy)
    return {"trained": True, "accuracy": accuracy}

# ---------------------------
# Component: deploy_model_to_gcs_v5
# ---------------------------
@component(base_image="python:3.10")
def deploy_model_to_gcs_v5(model: Input[Model], bucket_name: str) -> dict:
    import os
    from datetime import datetime
    if not os.listdir(model.path) or "placeholder.txt" in os.listdir(model.path):
        return {"uploaded": False, "gcs_path": "none"}
    model_gcs_path = f"gs://{bucket_name}/models/retrained_bert_v5_{int(datetime.now().timestamp())}"
    os.system(f"gsutil -m rsync -r {model.path} {model_gcs_path}")
    return {"uploaded": True, "gcs_path": model_gcs_path}

# ---------------------------
# Pipeline definition
# ---------------------------
@pipeline(name="bert-sentiment-retraining-pipeline-v5-final") # New name to break cache
def bert_retraining_pipeline_v5_final(
    project_id: str = "ms-gcu-dissertation",
    dataset_name: str = "bert_predictions",
    days_back: int = 30,
    base_model_path: str = "distilbert-base-uncased",
    bucket_name: str = "ms-gcu-dissertation-bert-predictions",
):
    extract_task = extract_training_data_v5(project_id=project_id, dataset_name=dataset_name, days_back=days_back)
    prepare_task = prepare_training_data_v5(input_data=extract_task.outputs["output_data"])
    retrain_task = retrain_bert_model_v5(
        train_data=prepare_task.outputs["train_data"],
        val_data=prepare_task.outputs["val_data"],
        base_model_path=base_model_path,
    )
    deploy_task = deploy_model_to_gcs_v5(model=retrain_task.outputs["model_output"], bucket_name=bucket_name)

# ---------------------------
# Compile entry
# ---------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=bert_retraining_pipeline_v5_final,
        package_path="bert_retraining_pipeline_v5-final.yaml", # New file name
    )
    print("✅ Pipeline compiled to bert_retraining_pipeline_v5-final.yaml")