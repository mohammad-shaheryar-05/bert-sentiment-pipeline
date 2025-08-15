from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Model Loading ---
# This section will run once when the container starts.
MODEL_PATH = "."
tokenizer = None
model = None

try:
    logger.info("Starting model load...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval() # Set the model to evaluation mode
    logger.info("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"❌ FATAL: Model loading failed: {str(e)}")
    # If the model fails to load, the container will crash and restart, which is the desired behavior.
    # This prevents it from trying to serve requests with a broken model.
    tokenizer = None
    model = None
# --- End Model Loading ---


# THIS IS THE KEY FIX: The route now accepts POST requests at the root URL "/"
@app.route("/", methods=['POST'])
def predict():
    start_time = time.time()
    
    # Check if the model was loaded successfully on startup
    if model is None or tokenizer is None:
        logger.error("Model is not loaded. Cannot process request.")
        return jsonify({"error": "Model is not available"}), 503 # 503 Service Unavailable

    try:
        data = request.get_json()
        if not data or "review_text" not in data:
            logger.warning("Request received with missing 'review_text'")
            return jsonify({"error": "Missing 'review_text' in request body"}), 400
        
        review_text = data.get("review_text", "")
        
        if not review_text.strip():
            logger.warning("Request received with empty 'review_text'")
            return jsonify({"error": "Empty review text provided"}), 400
        
        # Tokenize and predict
        inputs = tokenizer(
            review_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)

        # Map prediction to label
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = label_map.get(predicted_class_idx.item(), "Unknown")
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        logger.info(f"Prediction successful for review: '{review_text[:50]}...' -> {sentiment}")
        
        return jsonify({
            "review": review_text, 
            "sentiment": sentiment,
            "confidence": confidence.item(),
            "processing_time_ms": processing_time_ms
        })
    
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {str(e)}")
        return jsonify({"error": "Prediction failed due to an internal error"}), 500

if __name__ == "__main__":
    # This block is for local development and will not be used by Cloud Run.
    # Cloud Run uses a production WSGI server like Gunicorn.
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )