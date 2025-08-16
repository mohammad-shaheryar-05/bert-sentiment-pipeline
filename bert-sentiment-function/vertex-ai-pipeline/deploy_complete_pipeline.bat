@echo off
echo Deploying Complete BERT MLOps Pipeline
echo ========================================

:: --- Path Configuration ---
:: This script assumes it is being run from the 'vertex-ai-pipeline' directory.
:: It uses relative paths to find the other necessary files.

:: FIX: Corrected the path to point to the actual 'cloud-function' directory.
set FUNCTION_PATH=".\cloud-function"

:: FIX: Simplified path to use a relative path.
set DASHBOARD_PATH=".\monitoring_dashboard.json"

:: FIX: Corrected the path to point up one level to the 'tests' directory.
set TEST_SCRIPT_PATH="..\..\tests\test_bert_api.py"


:: 1. Deploy enhanced Cloud Function
echo  Deploying Cloud Function...
gcloud functions deploy bert-sentiment-processor ^
    --gen2 ^
    --runtime=python311 ^
    --region=us-central1 ^
    --source=%FUNCTION_PATH% ^
    --entry-point=bert_sentiment_processor ^
    --trigger-topic=bert-sentiment-requests ^
    --timeout=540 ^
    --memory=512MB ^
    --max-instances=10 ^
    --set-env-vars="BERT_API_URL=https://bert-sentiment-service-121496194098.us-central1.run.app,BUCKET_NAME=ms-gcu-dissertation-bert-predictions,DRIFT_THRESHOLD=0.15"

if %ERRORLEVEL% neq 0 (
    echo ❌ ERROR: Cloud Function deployment failed.
    exit /b %ERRORLEVEL%
)

:: 2. Setup BigQuery
echo 📊 Setting up BigQuery...
bq mk --dataset ms-gcu-dissertation:bert_predictions
if %ERRORLEVEL% neq 0 (
    echo ℹ️ INFO: BigQuery dataset already exists or another error occurred.
)

:: 3. Create monitoring dashboard
echo 📈 Creating monitoring dashboard...
gcloud monitoring dashboards create --config-from-file=%DASHBOARD_PATH%
if %ERRORLEVEL% neq 0 (
    echo ℹ️ INFO: Monitoring dashboard creation failed or already exists.
)

:: 4. Run integration tests
if exist %TEST_SCRIPT_PATH% (
    echo 🧪 Running integration tests...
    py %TEST_SCRIPT_PATH%
    if %ERRORLEVEL% neq 0 (
        echo ❌ ERROR: Integration tests failed.
        exit /b %ERRORLEVEL%
    )
) else (
    echo ⚠️ WARNING: Test script not found at %TEST_SCRIPT_PATH%, skipping tests.
)

:: 5. Send test data
echo 📤 Sending test data...
gcloud pubsub topics publish bert-sentiment-requests ^
    --message="{\"review_text\": \"Complete pipeline test - this is amazing!\", \"request_id\": \"complete-test-001\"}"

if %ERRORLEVEL% neq 0 (
    echo ❌ ERROR: Failed to publish test data.
    exit /b %ERRORLEVEL%
)

echo ✅ Complete pipeline deployment finished!
echo 🌐 Dashboard: https://console.cloud.google.com/monitoring/dashboards
echo 📊 BigQuery: https://console.cloud.google.com/bigquery
echo ⚡ Cloud Functions: https://console.cloud.google.com/functions