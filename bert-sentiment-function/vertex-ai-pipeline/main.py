import os
import json
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from google.cloud import storage, aiplatform
from google.api_core.exceptions import NotFound

# ---- Config via env vars ----
PROJECT_ID = os.environ["GCP_PROJECT"]
REGION = os.environ.get("REGION", "europe-west2")
BUCKET_NAME = os.environ["BUCKET_NAME"]                 # e.g. amazon-review-data
PREDICTIONS_PREFIX = os.environ.get("PREDICTIONS_PREFIX", "predictions")
BASELINE_JSON = os.environ.get("BASELINE_JSON", '{"Positive":0.6,"Neutral":0.2,"Negative":0.2}')
THRESHOLD = float(os.environ.get("DRIFT_TVD_THRESHOLD", "0.15"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "500"))
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "24"))

PIPELINE_SPEC_URI = os.environ["PIPELINE_SPEC_URI"]     # e.g. gs://.../bert_retrain.json
PIPELINE_ROOT = os.environ["PIPELINE_ROOT"]             # e.g. gs://.../pipeline_root/
PIPELINE_DISPLAY_NAME_PREFIX = os.environ.get("PIPELINE_DISPLAY_NAME_PREFIX", "bert-retrain")

storage_client = storage.Client()

def _list_recent_prediction_blobs(bucket_name: str, prefix: str, hours: int) -> List[storage.Blob]:
    """List blobs in the last N hours under the predictions prefix."""
    bucket = storage_client.bucket(bucket_name)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    blobs = bucket.list_blobs(prefix=prefix)
    recent = []
    for b in blobs:
        # Some Storage backends may not populate time_created in list; fetch metadata if needed.
        # Ensure we have time_created:
        if not b.time_created:
            try:
                b.reload()
            except NotFound:
                continue
        if b.time_created and b.time_created >= cutoff:
            recent.append(b)
    return recent

def _load_sentiments(blobs: List[storage.Blob]) -> List[str]:
    sentiments = []
    for b in blobs:
        try:
            content = b.download_as_bytes()
            doc = json.loads(content.decode("utf-8"))
            # Expecting {"prediction": {"predicted_sentiment": "Positive"}, "input": {...}}
            pred = doc.get("prediction", {})
            label = pred.get("predicted_sentiment")
            if label:
                sentiments.append(label)
        except Exception as e:
            # Skip bad files
            print(f"Skip {b.name}: {e}")
    return sentiments

def _distribution(sentiments: List[str]) -> Dict[str, float]:
    total = len(sentiments)
    if total == 0:
        return {}
    counts = {}
    for s in sentiments:
        counts[s] = counts.get(s, 0) + 1
    return {k: v / total for k, v in counts.items()}

def _tvd(p: Dict[str, float], q: Dict[str, float], classes=("Positive","Neutral","Negative")) -> float:
    # Total Variation Distance: 0.5 * sum |p_i - q_i|
    return 0.5 * sum(abs(p.get(c, 0.0) - q.get(c, 0.0)) for c in classes)

def _trigger_vertex_pipeline(dist_now: Dict[str, float], sample_n: int):
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_ROOT)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    display_name = f"{PIPELINE_DISPLAY_NAME_PREFIX}-{ts}"

    # You can pass parameters your pipeline expects here:
    params = {
        "bucket": BUCKET_NAME,
        "predictions_prefix": PREDICTIONS_PREFIX,
        "sample_count": sample_n,
        "observed_distribution": json.dumps(dist_now),
        # Add any other training/eval params your pipeline spec expects:
        # "train_data_uri": f"gs://{BUCKET_NAME}/training_data/...",
        # "learning_rate": 2e-5,
        # ...
    }

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=PIPELINE_SPEC_URI,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=params,
        enable_caching=False,
    )
    job.submit()  # non-blocking
    print(f"✅ Triggered Vertex AI Pipeline: {display_name}")

def drift_check_pubsub(event, context):
    """Pub/Sub-triggered (Gen2) drift detection."""
    # 1) Gather recent predictions
    blobs = _list_recent_prediction_blobs(BUCKET_NAME, PREDICTIONS_PREFIX, LOOKBACK_HOURS)
    sentiments = _load_sentiments(blobs)
    n = len(sentiments)
    print(f"Found {n} predictions in last {LOOKBACK_HOURS}h")

    if n < MIN_SAMPLES:
        print(f"Not enough samples (n={n} < {MIN_SAMPLES}); skip retrain.")
        return

    dist_now = _distribution(sentiments)
    baseline = json.loads(BASELINE_JSON)

    tvd_val = _tvd(dist_now, baseline)
    print(f"Current distribution: {dist_now} | Baseline: {baseline} | TVD={tvd_val:.4f}")

    # 2) Decide on drift
    if tvd_val >= THRESHOLD:
        print("⚠️ Drift detected; triggering Vertex AI Pipeline.")
        _trigger_vertex_pipeline(dist_now, n)
    else:
        print("✅ No drift (below threshold); no action.")
