#!/usr/bin/env python

import pandas as pd
import numpy as np
import wals
import datetime
import os
import sh

import tensorflow as tf
from scipy.sparse import coo_matrix

# default hyperparameters
DEFAULT_HYPERPARAMS = {
    'weights': True,
    'latent_factors': 5,
    'num_iters': 20,
    'regularization': 0.07,
    'unobs_weight': 0.01,
    'wt_type': 0,
    'feature_wt_factor': 130.0,
    'feature_wt_exp': 0.08
}

def create_test_and_train_sets(data, test_percentage):
    """Create test and train sets, for different input data types.
    Args:
        data: dataframe with all data

    Returns:
        array of entity IDs
        array of target entity IDs
        sparse coo_matrix for training
        sparse coo_matrix for test
    Raises:
        ValueError: if invalid data_type is supplied
    """

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
        
    tr_sparse, test_sparse = create_sparse_train_and_test(ratings, n_entity, n_target_entity, test_percentage)

    return ratings[:, 0], ratings[:, 1], tr_sparse, test_sparse
    

def create_sparse_train_and_test(ratings, n_entity, n_target_entity, test_percentage):
  """Given ratings, create sparse matrices for train and test sets.
  Args:
    ratings:    list of ratings tuples  (u, i, r)
    n_entity:   number of entities
    n_target_entity:  number of target items
    test_percentage: percentage of data used to test the model
  Returns:
     train, test sparse matrices in scipy coo_matrix format.
  """
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


def train_model(args, tr_sparse):
  """Instantiate WALS model and use "simple_train" to factorize the matrix.
  Args:
    args: training args containing hyperparams
    tr_sparse: sparse training matrix
  Returns:
     the row and column factors in numpy format.
  """
  dim = args['latent_factors']
  num_iters = args['num_iters']
  reg = args['regularization']
  unobs = args['unobs_weight']
  wt_type = args['wt_type']
  feature_wt_exp = args['feature_wt_exp']
  obs_wt = args['feature_wt_factor']

  tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

  # generate model
  input_tensor, row_factor, col_factor, model = wals.wals_model(tr_sparse,dim,reg,unobs,args['weights'],wt_type,feature_wt_exp,obs_wt)

  # factorize matrix
  session = wals.simple_train(model, input_tensor, num_iters)

  tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

  # evaluate output factor matrices
  output_row = row_factor.eval(session=session)
  output_col = col_factor.eval(session=session)

  # close the training session now that we've evaluated the output
  session.close()

  return output_row, output_col


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


def generate_recommendations(entity_index, target_entity_rated, row_factor, col_factor, k):
  """Generate recommendations for a user.
  Args:
    entity_index: the row index of the user in the ratings matrix,
    target_entity_rated: the list of item indexes (column indexes in the ratings matrix)
      previously rated by that user (which will be excluded from the
      recommendations),
    row_factor: the row factors of the recommendation model
    col_factor: the column factors of the recommendation model
    k: number of recommendations requested
  Returns:
    list of k item indexes with the predicted highest rating,
    excluding those that the user has already rated
  """

  # bounds checking for args
  assert (row_factor.shape[0] - len(target_entity_rated)) >= k

  # retrieve entity factor
  entity_f = row_factor[entity_index]

  # dot product of item factors with user factor gives predicted ratings
  pred_ratings = col_factor.dot(entity_f)

  # find candidate recommended item indexes sorted by predicted rating
  k_r = k + len(target_entity_rated)
  candidate_items = np.argsort(pred_ratings)[-k_r:]

  # remove previously rated items and take top k
  recommended_items = [i for i in candidate_items if i not in target_entity_rated]
  recommended_items = recommended_items[-k:]

  # flip to sort highest rated first
  recommended_items.reverse()

  return recommended_items