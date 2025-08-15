from flask import Flask, request, jsonify
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and preprocessors
model = tf.keras.models.load_model("lstm_sentiment_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review_text = data.get("review_text", "")
    
    # Preprocess
    seq = tokenizer.texts_to_sequences([review_text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    
    # Predict
    prediction = model.predict(padded)
    sentiment_label = label_encoder.inverse_transform([prediction.argmax(axis=1)[0]])[0]
    
    return jsonify({"review": review_text, "predicted_sentiment": sentiment_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
