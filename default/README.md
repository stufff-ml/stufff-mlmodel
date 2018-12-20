
## Run locally

python test.py --job-dir ../../data --model-id foo1233.default --model-rev 2

## Train locally

```shell
gcloud ml-engine local train --job-dir $OUTPUT_PATH --module-name trainer.task --package-path trainer/ -- --model-id foo1233.default --model-rev 2
```

## Train on Google ML Engine

```shell

REGION=europe-west1
BUCKET_NAME=models.stufff.review

MODEL_ID=foo1233.default
MODEL_REV=2

JOB_ID=$(date +%s)
JOB_NAME=`echo $MODEL_ID.$MODEL_REV | tr . _`
OUTPUT_PATH=gs://$BUCKET_NAME/$MODEL_ID/"$JOB_NAME"_"$JOB_ID"

gcloud ml-engine local train --job-dir $OUTPUT_PATH --module-name trainer.task --package-path trainer/ -- --model-id foo1233.default --model-rev 2
````

or

```shell

gcloud ml-engine jobs submit training "$JOB_NAME"_"$JOB_ID" --job-dir $OUTPUT_PATH --module-name trainer.task --package-path trainer/ --region $REGION -- --model-id foo1233.default --model-rev 2

````



#OUTPUT_PATH=gs://$BUCKET_NAME/$MODEL_ID/output/$JOB_ID
