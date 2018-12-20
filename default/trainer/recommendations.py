#!/usr/bin/env python

import pandas as pd
import numpy as np


def single_recommendation(entity_index, exclude, row_factor, col_factor, k):
  """Generate recommendations for a user.
  Args:
    entity_index: the row index of the user in the ratings matrix,
    exclude: the list of item indexes (column indexes in the ratings matrix)
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
  assert (row_factor.shape[0] - len(exclude)) >= k

  # retrieve entity factor
  entity_f = row_factor[entity_index]

  # dot product of item factors with user factor gives predicted ratings
  pred_ratings = col_factor.dot(entity_f)

  # find candidate recommended item indexes sorted by predicted rating
  k_r = k + len(exclude)
  candidate_items = np.argsort(pred_ratings)[-k_r:]

  # remove previously rated items and take top k
  recommended_items = [i for i in candidate_items if i not in exclude]
  recommended_items = recommended_items[-k:]

  # flip to sort highest rated first
  recommended_items.reverse()

  return recommended_items


def batch_recommendation(data, row_factor, col_factor, k):

  unique_entity = np.unique(data.entity_id.values)
  unique_target_entity= np.unique(data.target_entity_id.values)

  #recommended_items = pd.DataFrame(index=unique_entity, columns=['r'])
  recommended_items = pd.DataFrame(columns=np.zeros(shape=k,dtype=np.int32))

  for id in unique_entity:
    item_idx = np.searchsorted(unique_entity, id)
    already_rated = data[data.entity_id.isin([id])].target_entity_id
    already_rated_idx = [np.searchsorted(unique_target_entity, i)
                          for i in already_rated]

    recs_idx = single_recommendation(item_idx, already_rated_idx, row_factor, col_factor, k)
    recs = [unique_target_entity[i] for i in recs_idx]

    recommended_items.loc[id] = recs

  return recommended_items