import os
from google.cloud import pubsub_v1
import pandas as pd
import time
import json
from google.api_core.exceptions import GoogleAPICallError

# Configuration
PROJECT_ID = "ms-gcu-dissertation"  
TOPIC_ID = "amazon-reviews-topic"
CSV_FILE = "preprocessed_amazon_reviews.csv"  # Ensure this file exists in the same directory or provide full path

def resolve_csv_path(csv_file):
    """Resolve the absolute path to the CSV file."""
    if not os.path.exists(csv_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_file} or {csv_path}")
        return csv_path
    return csv_file

def publish_messages(publisher, topic_path, df):
    """Publish DataFrame rows as Pub/Sub messages."""
    for index, row in df.iterrows():
        try:
            message_json = json.dumps({
                "review_text": row.get("cleaned_text", ""),
                "rating": row.get("reviews.rating", 0),
                "sentiment": row.get("sentiment", "neutral")
            }).encode("utf-8")
            
            future = publisher.publish(topic_path, message_json)
            future.result()  # Wait for the publish to complete
            print(f"✅ Published message {index+1}/{len(df)} (ID: {future.result()})")
            
        except (KeyError, json.JSONEncodeError) as e:
            print(f"❌ Error processing row {index+1}: {e}")
        except GoogleAPICallError as e:
            print(f"❌ Pub/Sub API error: {e}")
            break  # Stop if publishing fails
            
        time.sleep(0.5)  # Throttle messages

def main():
    try:
        # Initialize Pub/Sub client
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
        
        # Load CSV with explicit path resolution
        csv_path = resolve_csv_path(CSV_FILE)
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} rows from {csv_path}")
        
        # Publish messages
        publish_messages(publisher, topic_path, df)
        
    except Exception as e:
        print(f"🔥 Critical error: {e}")
    finally:
        print("🏁 Publisher finished")

if __name__ == "__main__":
    main()