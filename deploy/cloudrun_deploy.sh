#!/usr/bin/env bash
# ==============================================================================
# Google Cloud Run Deployment Script for YogaSutra
# ==============================================================================
set -euo pipefail

# Configuration parameters
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-yogasutra-api}"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "=============================================================================="
echo "YogaSutra: Deploying to Google Cloud Run"
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Image:    ${IMAGE_TAG}"
echo "=============================================================================="

# Check for required API key
if [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "ERROR: GOOGLE_API_KEY environment variable is not set."
  echo "Please export GOOGLE_API_KEY='your-key' before running."
  exit 1
fi

# Build container using Google Cloud Build (or local docker push)
echo "==> Building container image via Google Cloud Build..."
gcloud builds submit --tag "${IMAGE_TAG}" .

# Deploy container to Google Cloud Run
echo "==> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_TAG}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 80 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY},YOGASUTRA_MODEL=${YOGASUTRA_MODEL:-gemini-2.5-flash},YOGASUTRA_LOG_LEVEL=INFO,ENABLE_GUARDRAILS=true"

# Fetch and display the deployed service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format 'value(status.url)')
echo "=============================================================================="
echo "Deployment successful!"
echo "YogaSutra Studio & API URL: ${SERVICE_URL}"
echo "Health Check: ${SERVICE_URL}/api/health"
echo "API Docs:     ${SERVICE_URL}/docs"
echo "=============================================================================="
