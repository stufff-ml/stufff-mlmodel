# stufff-mlmodel

Default models for the Stufff Machine Learning Engine

### Reference

* https://cloud.google.com/ml-engine/docs/tensorflow/how-tos

* https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview
* https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals

* https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction
* https://cloud.google.com/ml-engine/docs/tensorflow/training-overview 
* https://cloud.google.com/ml-engine/docs/tensorflow/samples 
* https://github.com/GoogleCloudPlatform/cloudml-samples/ 
* https://github.com/Azure/LearnAI-Bootcamp 
* https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html 
* https://towardsdatascience.com/how-to-train-machine-learning-models-in-the-cloud-using-cloud-ml-engine-3f0d935294b3 
* https://hackernoon.com/tensorrec-a-recommendation-engine-framework-in-tensorflow-d85e4f0874e8 

Datasets

Brazilian E-Commerce Public Dataset by Olist
https://www.kaggle.com/olistbr/brazilian-ecommerce#olist_order_items_dataset.csv

Online Retail Data Set from UCI ML repo
https://www.kaggle.com/jihyeseo/online-retail-data-set-from-uci-ml-repo

Retailrocket recommender system dataset
https://www.kaggle.com/retailrocket/ecommerce-dataset#events.csv

## Development

pip install -U pip
pip install --user --upgrade virtualenv

virtualenv stufff-env
source stufff-env/bin/activate

pip install -r requirements.txt


## Format

### Import

```csv
event,entity_type,entity_id,target_entity_type,target_entity_id,timestamp,properties
buy,user,121688,item,15335,1548705546,''
buy,user,599528,item,356475,1548705546,''
```

### Export

```csv
entity_id,entity_type,target_entity_type,values
1,user,item,"[99, 0.99982, 24, 0.99659, 8, 0.98458]"
```