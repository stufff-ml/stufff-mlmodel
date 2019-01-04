#!/usr/bin/env python

import numpy as np
import os
import sh


def export_model(job_id, export_location, entity_map, target_entity_map, row_factor, col_factor):
  
  model_dir = export_location
  
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(job_id)

  if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

  np.save(os.path.join(model_dir, 'entity'), entity_map)
  np.save(os.path.join(model_dir, 'target_entity'), target_entity_map)
  np.save(os.path.join(model_dir, 'row'), row_factor)
  np.save(os.path.join(model_dir, 'col'), col_factor)

  if gs_model_dir:
    sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)


def export_predictions(job_id, export_location, recs):

  model_dir = export_location
  
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(job_id)

  if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

  recs.to_csv(os.path.join(model_dir, 'recs.csv'),header=False,encoding='utf-8')

  if gs_model_dir:
    sh.gsutil('cp', '-r', os.path.join(model_dir, 'recs.csv'), gs_model_dir)

