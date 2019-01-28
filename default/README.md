
## Run locally

```shell
python task.py --job-dir ../../data --model-id foo1233.default --model-rev 2
````

## Train with Google ML Engine

### Setup

```shell

BUCKET_NAME=models.stufff.review
REGION=europe-west1
MODEL_NAME=default
PACKAGE=default-1
CLIENT_ID=foo123

TS=$(date +%s)
JOB_ID="$MODEL_NAME"_"$CLIENT_ID"_"$TS"
JOB_DIR=gs://$BUCKET_NAME/$CLIENT_ID/"$JOB_ID"
CALLBACK="http://localhost:8080/_i/1/callback/train?id=$CLIENT_ID&job=$JOB_ID"

```

### Locally

```shell

gcloud ml-engine local train \
  --module-name model.train \
  --job-dir $JOB_DIR \
  --package-path model/ \
  -- \
  --client-id $CLIENT_ID --model-name $MODEL_NAME --job-id $JOB_ID --callback $CALLBACK

```

### Cloud ML Engine

```shell

gcloud ml-engine jobs submit training "$JOB_NAME"_"$JOB_ID" \
  --module-name model.train \
  --job-dir $JOB_DIR \
  --package-path model/ \
  --region $REGION \
  --runtime-version 1.12 --python-version 2.7 \
  -- \
  --client-id $CLIENT_ID --model-name $MODEL_NAME --job-id $JOB_ID --callback $CALLBACK

```

#### With shared package

```shell

PACKAGE_PATH=gs://"$BUCKET_NAME/packages/$MODEL_PACKAGE/$MODEL_PACKAGE".tar.gz

gcloud ml-engine jobs submit training "$JOB_NAME"_"$JOB_ID" \
  --module-name model.task \
  --region $REGION \
  --job-dir $OUTPUT_PATH \
  --packages $PACKAGE_PATH \
  --runtime-version 1.12 --python-version 2.7 \
  -- \
  --model-id $MODEL_ID --model-rev $MODEL_REV

```
