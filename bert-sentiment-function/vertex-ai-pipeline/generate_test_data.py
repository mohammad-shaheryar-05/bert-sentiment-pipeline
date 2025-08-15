#!/usr/bin/env python3

import json
import subprocess
import random
import time
from datetime import datetime

# Sample reviews for testing
SAMPLE_REVIEWS = [
    ("This product is absolutely fantastic! I love it!", "positive"),
    ("Terrible quality, complete waste of money", "negative"), 
    ("It's okay, nothing special but does the job", "neutral"),
    ("Amazing service and fast delivery!", "positive"),
    ("Worst customer service ever experienced", "negative"),
    ("Average product, meets basic expectations", "neutral"),
    ("Outstanding quality, highly recommend!", "positive"),
    ("Broken on arrival, very disappointed", "negative"),
    ("Standard product, works as expected", "neutral"),
    ("Exceeded all my expectations!", "positive")
]

def send_prediction_request(review_text, expected_sentiment, request_id):
    """
    Sends a single prediction request to the Pub/Sub topic.
    NOTE: This version is adapted for Windows by correctly formatting the --message flag.
    """
    message_dict = {
        "review_text": review_text,
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "expected_sentiment": expected_sentiment
    }
    
    # Correctly format the message for Windows cmd.exe: wrap in quotes and escape internal quotes
    message_str = json.dumps(message_dict)

    cmd = [
        "gcloud", "pubsub", "topics", "publish", "bert-sentiment-requests",
        f"--message={message_str}"
    ]
    
    # Use shell=True on Windows for gcloud commands within subprocess
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        print(f"✅ Sent: {request_id} - {expected_sentiment}")
    else:
        print(f"❌ Failed: {request_id} - {result.stderr}")

def main():
    print("🚀 Generating test data for BERT pipeline...")
    
    for i in range(20):
        review, sentiment = random.choice(SAMPLE_REVIEWS)
        request_id = f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{i:03d}"
        
        send_prediction_request(review, sentiment, request_id)
        time.sleep(2)  # Wait 2 seconds between requests
    
    print("✅ Test data generation complete!")

if __name__ == "__main__":
    main()