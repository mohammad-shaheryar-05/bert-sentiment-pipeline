from locust import HttpUser, task, between
import json
import random

class BertAPIUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    host = "https://bert-sentiment-service-121496194098.us-central1.run.app"
    
    test_reviews = [
        "This product is amazing!",
        "Terrible quality, disappointed.",
        "Average product, nothing special.",
        "Outstanding service and delivery!",
        "Worst purchase ever made."
    ]
    
    def on_start(self):
        # Optional: Add authentication header if Cloud Run requires it
        self.headers = {
            "Content-Type": "application/json"
            # Uncomment and replace with valid ID token if authentication is needed
            # "Authorization": "Bearer YOUR_ID_TOKEN"
        }
    
    @task
    def predict_sentiment(self):
        review = random.choice(self.test_reviews)
        self.client.post("/predict", 
            json={"review_text": review},
            headers=self.headers)
    
    @task(1)  # Lower weight for health check
    def health_check(self):
        self.client.get("/")