import os
import sh
import numpy as np
import pandas as pd
from io import BytesIO
from google.cloud import storage


def read_dataframe(client_id, source_bucket, f):
  client = storage.Client()

  # Load the training data
  source_name = client_id + '/' + f

  source_bucket = client.get_bucket(source_bucket)
  source_blob = source_bucket.get_blob(source_name)
  content = source_blob.download_as_string()

  headers = ['event','entity_type','entity_id','target_entity_type','target_entity_id','timestamp','properties']
  header_types = {'entity_id':np.int32, 'target_entity_id':np.int32, 'timestamp':np.int32 }
  raw_df = pd.read_csv(BytesIO(content), names=headers, header=None, dtype=header_types)

  return raw_df


def write_dataframe(job_id, target_bucket, f, cols, label, recs):

  local_dir = '/tmp/{0}'.format(job_id)
  if not os.path.isdir(local_dir):
    os.makedirs(local_dir)
  
  # write the file to tmp 
  export_file = os.path.join(local_dir, f)
  recs.to_csv(export_file, header=True, columns=cols, index_label=label, encoding='utf-8')

  # upload the file
  sh.gsutil('cp', '-r', export_file, os.path.join(target_bucket, f))
