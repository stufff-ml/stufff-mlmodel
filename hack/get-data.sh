#!/bin/bash

echo "Get data to training local ML model"

MODEL_ID="foo1233.default"
MODEL_REV="2"

DATA_DIR="data"
BUCKET_NAME="models.stufff.review"

DATA_FILE="gs://$BUCKET_NAME/$MODEL_ID/$MODEL_ID.$MODEL_REV.csv"

gsutil -m cp $DATA_FILE $DATA_DIR
