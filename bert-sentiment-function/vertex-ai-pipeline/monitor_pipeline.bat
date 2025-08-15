@echo off
echo.
echo === BERT Pipeline Status ===
echo Date: %date% %time%
echo.

echo === Recent Predictions ===
bq query --use_legacy_sql=false --format=pretty "SELECT predicted_sentiment, COUNT(*) as count, AVG(confidence) as avg_confidence, MIN(timestamp) as earliest, MAX(timestamp) as latest FROM `ms-gcu-dissertation.bert_predictions.prediction_history` WHERE timestamp >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 24 HOUR) GROUP BY predicted_sentiment ORDER BY count DESC"
echo.

echo === Storage Usage ===
gsutil du -sh gs://ms-gcu-dissertation-bert-predictions/predictions/
echo.

echo === Recent Function Invocations ===
gcloud functions logs read bert-sentiment-processor --region=us-central1 --limit=5 --format="value(timestamp, textPayload)"