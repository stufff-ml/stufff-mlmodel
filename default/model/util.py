import os
import sh
from io import BytesIO


def read_dataframe():

def write_dataframe(job_id, f, export_location, cols, label, recs):

  local_dir = export_location
  
  remote_dir = None
  if export_location.startswith('gs://'):
    remote_dir = export_location
    local_dir = '/tmp/{0}'.format(job_id)

  if not os.path.isdir(local_dir):
    os.makedirs(local_dir)

  export_file = os.path.join(local_dir, f)
  recs.to_csv(export_file, header=True, columns=cols, index_label=label, encoding='utf-8')

  if remote_dir:
    sh.gsutil('cp', '-r', export_file, os.path.join(remote_dir, f))

