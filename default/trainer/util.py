#!/usr/bin/env python

import numpy as np
import os
import sh


def save_model(model_id, model_rev, output_dir, entity_map, target_entity_map, row_factor, col_factor):
  """Save the user map, item map, row factor and column factor matrices in numpy format.
  These matrices together constitute the "recommendation model."
  Args:
    entity_map:     entity map numpy array
    target_entity_map:     target_entity map numpy array
    row_factor:   row_factor numpy array
    col_factor:   col_factor numpy array
  """

  model_dir = os.path.join(output_dir, 'model')
  job_name = model_id + '_' + model_rev

  # if our output directory is a GCS bucket, write model files to /tmp,
  # then copy to GCS
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(job_name)

  if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

  np.save(os.path.join(model_dir, 'entity'), entity_map)
  np.save(os.path.join(model_dir, 'target_entity'), target_entity_map)
  np.save(os.path.join(model_dir, 'row'), row_factor)
  np.save(os.path.join(model_dir, 'col'), col_factor)

  if gs_model_dir:
    sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)


def save_recommendations(model_id, model_rev, output_dir, recs):
  model_dir = os.path.join(output_dir, 'model')
  job_name = model_id + '_' + model_rev

  # if our output directory is a GCS bucket, write model files to /tmp,
  # then copy to GCS
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(job_name)

  if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

  recs.to_csv(os.path.join(model_dir, 'recs.csv'),header=False,encoding='utf-8')

  if gs_model_dir:
    sh.gsutil('cp', '-r', os.path.join(model_dir, 'recs.csv'), gs_model_dir)

