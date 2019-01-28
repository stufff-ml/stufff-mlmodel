
import argparse
import os
from io import BytesIO
from google.cloud import storage
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix
import wals
import prediction as pre
import datetime
import sh

def train(params):
    print(" --> PARAMS")
    print(params)
    print('')

    client = storage.Client()

    # Load the training data
    source_name = params.client_id + '/buy.csv'
    source_bucket = client.get_bucket(params.source_bucket)
    source_blob = source_bucket.get_blob(source_name)
    content = source_blob.download_as_string()

    headers = ['event','entity_type','entity_id','target_entity_type','target_entity_id','timestamp','properties']
    header_types = {'entity_id':np.int32, 'target_entity_id':np.int32, 'timestamp':np.int32 }
    data = pd.read_csv(BytesIO(content), names=headers, header=None, dtype=header_types)

    # create training and test sets
    entity_map, target_entity_map, training_sparse, test_sparse = create_data_sets(data, params.test_percentage)

    # generate model
    tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    input_tensor, row_factor, col_factor, model = wals.wals_model(training_sparse,
      params.latent_factors,
      params.regularization,
      params.unobs_weight,
      params.weights,
      params.wt_type,
      params.feature_wt_exp,
      params.feature_wt_factor)

    # factorize matrix
    session = wals.simple_train(model, input_tensor, params.num_iters)

    # evaluate output factor matrices
    output_row = row_factor.eval(session=session)
    output_col = col_factor.eval(session=session)

    tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    # close the training session now that we've evaluated the output
    session.close()
    
    # batch predictions for all entity IDs
    predictions = pre.batch_predictions(data, output_row, output_col, params.predict_batch_size)

    # save the model data and predictions
    export_model(params.job_id, params.job_dir, entity_map, target_entity_map, output_row, output_col)
    export_predictions(params.job_id, params.job_dir, predictions)

    # calculate the training accuracy
    train_rmse = wals.get_rmse(output_row, output_col, training_sparse)
    test_rmse = wals.get_rmse(output_row, output_col, test_sparse)
    print(' --> Training RMSE = %.2f' % train_rmse)
    print(' --> Test RMSE = %.2f' % test_rmse)





def create_data_sets(data, test_percentage):
    
    entity = data.entity_id.values
    unique_entity = np.unique(entity)

    target_entity = data.target_entity_id.values
    unique_target_entity= np.unique(target_entity)

    n_entity = unique_entity.shape[0]
    n_target_entity = unique_target_entity.shape[0]

    max_entity = unique_entity[-1]
    max_target_entity = unique_target_entity[-1]

    ratings = data.as_matrix(['entity_id', 'target_entity_id','properties'])
    ratings[:, 2] = 1.0
    
    if n_entity != max_entity or n_target_entity != max_target_entity:
        z = np.zeros(max_entity+1, dtype=int)
        z[unique_entity] = np.arange(n_entity)
        u_r = z[entity]

        z = np.zeros(max_target_entity+1, dtype=int)
        z[unique_target_entity] = np.arange(n_target_entity)
        i_r = z[target_entity]

        # construct the ratings set from the two stacks
        ratings[:, 0] = u_r
        ratings[:, 1] = i_r
    else:
        # deal with 1-based user indices
        ratings[:, 0] -= 1
        ratings[:, 1] -= 1
        
    tr_sparse, test_sparse = create_sparse_data_sets(ratings, n_entity, n_target_entity, test_percentage)

    return ratings[:, 0], ratings[:, 1], tr_sparse, test_sparse
    

def create_sparse_data_sets(ratings, n_entity, n_target_entity, test_percentage):
  
  # pick a random test set of entries, sorted ascending
  test_set_size = int(len(ratings) * test_percentage)
  test_set_idx = np.random.choice(xrange(len(ratings)), size=test_set_size, replace=False)
  test_set_idx = sorted(test_set_idx)

  # sift ratings into train and test sets
  ts_ratings = ratings[test_set_idx]
  tr_ratings = np.delete(ratings, test_set_idx, axis=0)

  # create training and test matrices as coo_matrix's
  u_tr, i_tr, r_tr = zip(*tr_ratings)
  tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_entity, n_target_entity))

  u_ts, i_ts, r_ts = zip(*ts_ratings)
  test_sparse = coo_matrix((r_ts, (u_ts, i_ts)), shape=(n_entity, n_target_entity))

  return tr_sparse, test_sparse


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

