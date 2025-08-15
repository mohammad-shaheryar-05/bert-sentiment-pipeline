import google.cloud.aiplatform as aip
import os

# --- Your Project Configuration ---
PROJECT_ID = "ms-gcu-dissertation"
REGION = "us-central1"
PIPELINE_ROOT = "gs://ms-gcu-dissertation-bert-predictions/pipeline-artifacts" # A folder in your GCS bucket for pipeline outputs
PIPELINE_DEFINITION_FILE = "bert_retraining_pipeline.yaml"
DISPLAY_NAME = "bert-retraining-run-from-python"

# --- Main Script ---
print(f"Initializing connection to Vertex AI for project '{PROJECT_ID}' in '{REGION}'...")

# Initialize the Vertex AI SDK
aip.init(project=PROJECT_ID, location=REGION)

print(f"Loading pipeline definition from: {PIPELINE_DEFINITION_FILE}")

# Create a pipeline job object from your compiled YAML file
pipeline_job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path=PIPELINE_DEFINITION_FILE,
    pipeline_root=PIPELINE_ROOT,
    enable_caching=True
)

print("Submitting the pipeline job to Vertex AI...")

# Submit the pipeline job to run on Vertex AI
# The 'sync=False' means the script will finish right away and not wait for the pipeline to complete.
pipeline_job.run(sync=False)

print("\n--- SUCCESS! ---")
print(f"Pipeline job '{pipeline_job.display_name}' has been submitted.")
print("You can monitor its progress in the Google Cloud Console at this URL:")
print(f"https://console.cloud.google.com/vertex-ai/pipelines/runs/{pipeline_job.name}?project={PROJECT_ID}")