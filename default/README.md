
## Train locally

```shell
gcloud ml-engine local train --module-name trainer.task --package-path trainer/ -- --model-id foo1233.default --model-rev 2
```

## Train on Google ML Engine

```shell

MODEL_ID=foo1233.default
MODEL_REV=2
JOB_NAME=foo1233_default_2

REGION=europe-west1
BUCKET_NAME=models.stufff.review

JOB_ID=$(date +%s)
OUTPUT_PATH=gs://$BUCKET_NAME/$MODEL_ID/output/$JOB_ID
gcloud ml-engine jobs submit training "$JOB_NAME"_"$JOB_ID" --job-dir $OUTPUT_PATH --module-name trainer.task --package-path trainer/ --region $REGION -- --model-id foo1233.default --model-rev 2

````
