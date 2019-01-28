
## Run locally

```shell
python task.py --job-dir ../../data --model-id foo1233.default --model-rev 2
````

## Train with Google ML Engine

### Setup

```shell

REGION=europe-west1
BUCKET_NAME=models.stufff.review

MODEL_ID=foo1233.default
MODEL_REV=2
MODEL_PACKAGE=default-1

````

### Locally

```shell

JOB_ID=$(date +%s)
JOB_NAME=`echo $MODEL_ID.$MODEL_REV | tr . _`
OUTPUT_PATH=gs://$BUCKET_NAME/$MODEL_ID/"$JOB_NAME"_"$JOB_ID"

gcloud ml-engine local train --job-dir $OUTPUT_PATH --module-name model.train --package-path model/ -- --model-id $MODEL_ID --model-rev $MODEL_REV

````

### Cloud ML Engine

```shell

JOB_ID=$(date +%s)
JOB_NAME=`echo $MODEL_ID.$MODEL_REV | tr . _`
OUTPUT_PATH=gs://$BUCKET_NAME/$MODEL_ID/"$JOB_NAME"_"$JOB_ID"

gcloud ml-engine jobs submit training "$JOB_NAME"_"$JOB_ID" \
  --module-name model.task \
  --region $REGION \
  --job-dir $OUTPUT_PATH \
  --package-path model/ \
  --runtime-version 1.12 --python-version 2.7 \
  -- \
  --model-id $MODEL_ID --model-rev $MODEL_REV

````

#### With shared package

```shell

JOB_ID=$(date +%s)
JOB_NAME=`echo $MODEL_ID.$MODEL_REV | tr . _`
OUTPUT_PATH=gs://$BUCKET_NAME/$MODEL_ID/"$JOB_NAME"_"$JOB_ID"
PACKAGE_PATH=gs://"$BUCKET_NAME/packages/$MODEL_PACKAGE/$MODEL_PACKAGE".tar.gz

gcloud ml-engine jobs submit training "$JOB_NAME"_"$JOB_ID" \
  --module-name model.task \
  --region $REGION \
  --job-dir $OUTPUT_PATH \
  --packages $PACKAGE_PATH \
  --runtime-version 1.12 --python-version 2.7 \
  -- \
  --model-id $MODEL_ID --model-rev $MODEL_REV

````
