#!/bin/bash


BUCKET_NAME="models.stufff.review"

source stufff-env/bin/activate

function package_and_upload {
  MODEL=$1
  VERSION=$2
  BUCKET=$3

  MODEL_NAME=$MODEL-$VERSION

  cwd=$(pwd)

  echo " --> Packaging model '$MODEL_NAME'"

  cd $MODEL
  python setup.py sdist > /dev/null

  PACKAGE="$MODEL_NAME".tar.gz
  UPLOAD_LOCATION="gs://$BUCKET/packages/$MODEL_NAME/"

  echo " --> Uploading model '$MODEL_NAME'"
  gsutil cp dist/$PACKAGE $UPLOAD_LOCATION

  echo " --> Cleanup"

  rm -rf dist
  rm -rf $MODEL.egg-info

  cd $cwd
}

echo " --> Package and upload models"
echo ""

# Package the models
package_and_upload "default" "1" $BUCKET_NAME
