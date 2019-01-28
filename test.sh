#!/bin/bash

source stufff-env/bin/activate

MODEL_BUCKET=models.stufff.review
DATA_BUCKET=exports.stufff.review
DATA_DIR="data"

REGION=europe-west1
MODEL_NAME=default
PACKAGE=default-1

function train() {

  cwd=$(pwd)
  cd default

  CLIENT_ID=$1

  TS=$(date +%s)
  JOB_ID="$MODEL_NAME"_"$CLIENT_ID"_"$TS"
  JOB_DIR=gs://$MODEL_BUCKET/$CLIENT_ID/"$JOB_ID"
  CALLBACK="http://localhost:8080/_i/1/callback/train?id=$CLIENT_ID&job=$JOB_ID"

  gcloud ml-engine local train \
    --module-name model.train \
    --job-dir $JOB_DIR \
    --package-path model/ \
    -- \
    --client-id $CLIENT_ID --model-name $MODEL_NAME --job-id $JOB_ID --callback $CALLBACK
    
  cd $cwd
}

function get_data() {
  CLIENT_ID=$1
  DATA_FILE="gs://$DATA_BUCKET/$CLIENT_ID/buy.csv"

  gsutil -m cp $DATA_FILE $DATA_DIR
}

get_data $1
train $1