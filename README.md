# stufff-mlmodel
Default model for the Stufff Machine Learning Engine

GOOGLE_APPLICATION_CREDENTIALS=service_account.json

pip install -U pip
pip install --user --upgrade virtualenv

virtualenv stufff-env
source stufff-env/bin/activate

pip install -r requirements.txt

### Reference

* https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview
* https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals

* https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction
* https://cloud.google.com/ml-engine/docs/tensorflow/training-overview 
* https://cloud.google.com/ml-engine/docs/tensorflow/samples 
* https://github.com/GoogleCloudPlatform/cloudml-samples/ 
* https://github.com/Azure/LearnAI-Bootcamp 
