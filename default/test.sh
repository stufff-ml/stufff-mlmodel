#!/bin/bash

BUCKET_NAME=models.stufff.review
REGION=europe-west1
MODEL_NAME=default
PACKAGE=default-1
CLIENT_ID=foo123

TS=$(date +%s)
JOB_ID="$MODEL_NAME"_"$CLIENT_ID"_"$TS"
JOB_DIR=gs://$BUCKET_NAME/$CLIENT_ID/"$JOB_ID"
CALLBACK="http://localhost:8080/_i/1/callback/train?id=$CLIENT_ID&job=$JOB_ID"

gcloud ml-engine local train \
  --module-name model.train \
  --job-dir $JOB_DIR \
  --package-path model/ \
  -- \
  --client-id $CLIENT_ID --model-name $MODEL_NAME --job-id $JOB_ID --callback $CALLBACK
